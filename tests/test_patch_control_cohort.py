from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from structxai.patch_control_cohort import (
    PatchControlCohortError,
    PatchControlCohortPolicy,
    PatchControlCohortRejected,
    PatchControlTask,
    audit_patch_control_cohort,
    require_patch_control_cohort,
)
from structxai.patch_dose_response import PatchSweepError, PatchSweepPoint, PatchSweepResult


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _sweep(source: str, target: str, curves: dict[int, tuple[float, ...]]) -> PatchSweepResult:
    target_margin = -1.0
    points = tuple(
        PatchSweepPoint(
            layer=layer,
            alpha=alpha,
            positive_score=target_margin + recovery * 2.0,
            negative_score=0.0,
            margin=target_margin + recovery * 2.0,
            margin_delta=recovery * 2.0,
            normalized_recovery=recovery,
        )
        for layer, recoveries in curves.items()
        for alpha, recovery in zip((0.0, 0.5, 1.0), recoveries, strict=True)
    )
    return PatchSweepResult(
        schema_version=1,
        model_revision="sha256:cohort-model",
        source_input_sha256=_digest(source),
        target_input_sha256=_digest(target),
        sweep_spec_sha256="c" * 64,
        policy_sha256="d" * 64,
        source_margin=1.0,
        target_margin=target_margin,
        repeated_target_margin=target_margin,
        model_calls=3 + len(points),
        points=points,
        evidence_sha256="",
    ).with_digest()


def _task(index: int, *, specific: bool = True) -> PatchControlTask:
    target = f"target-{index}"
    treatment = _sweep(
        f"treatment-{index}",
        target,
        {0: (0.0, 0.50, 1.0), 2: (0.0, 0.35, 0.70)},
    )
    strongest = 0.10 if specific else 0.90
    controls = {
        f"null-{control}": _sweep(
            f"control-{index}-{control}",
            target,
            {
                0: (0.0, strongest / 2.0, strongest),
                2: (0.0, strongest / 3.0, strongest * 0.8),
            },
        )
        for control in range(3)
    }
    return PatchControlTask(treatment=treatment, controls=controls)


def _cohort(*, failing: set[int] | None = None) -> dict[str, PatchControlTask]:
    failures = failing or set()
    return {f"task-{index}": _task(index, specific=index not in failures) for index in range(8)}


def test_accepts_fixed_cohort_with_simultaneous_layer_bounds() -> None:
    audit = require_patch_control_cohort(_cohort())

    assert audit.status == "accepted"
    assert audit.bound_method == "wilson-bonferroni-one-sided/v1"
    assert audit.hypothesis_count == 2
    assert audit.per_hypothesis_alpha == pytest.approx(0.025)
    assert audit.cohort_specific_layers == 2
    assert all(layer.specific_tasks == 8 for layer in audit.layers)
    assert all(layer.simultaneous_lower_bound > 0.5 for layer in audit.layers)
    assert len(audit.evidence_sha256) == 64


def test_rejects_when_simultaneous_lower_bound_misses_policy() -> None:
    audit = audit_patch_control_cohort(_cohort(failing={0, 1, 2, 3}))

    assert audit.status == "rejected"
    assert audit.cohort_specific_layers == 0
    assert audit.finding_codes == ("COHORT_SPECIFIC_LAYER_COUNT_BELOW_MINIMUM",)
    assert all(
        layer.finding_codes == ("SPECIFIC_TASK_FRACTION_LOWER_BOUND_BELOW_MINIMUM",) for layer in audit.layers
    )
    with pytest.raises(PatchControlCohortRejected, match="PATCH_CONTROL_COHORT_REJECTED"):
        require_patch_control_cohort(_cohort(failing={0, 1, 2, 3}))


def test_stricter_prespecified_fraction_can_reject_all_observed_successes() -> None:
    policy = PatchControlCohortPolicy(min_specific_task_fraction_lower_bound=0.70)

    audit = audit_patch_control_cohort(_cohort(), cohort_policy=policy)

    assert all(layer.observed_specific_fraction == 1.0 for layer in audit.layers)
    assert all(layer.simultaneous_lower_bound < 0.70 for layer in audit.layers)
    assert audit.status == "rejected"


def test_evidence_is_order_independent_and_omits_raw_task_and_control_ids() -> None:
    cohort = _cohort()
    first = audit_patch_control_cohort(cohort)
    second = audit_patch_control_cohort(dict(reversed(tuple(cohort.items()))))
    serialized = json.dumps(first.as_dict(), sort_keys=True)

    assert first == second
    assert "task-0" not in serialized
    assert "null-0" not in serialized
    assert len(first.tasks[0].task_id_sha256) == 64


def test_revalidates_child_sweeps_instead_of_trusting_task_container() -> None:
    cohort = _cohort()
    task = cohort["task-0"]
    cohort["task-0"] = replace(
        task,
        treatment=replace(task.treatment, evidence_sha256="0" * 64),
    )

    with pytest.raises(PatchSweepError, match="SWEEP_EVIDENCE_DIGEST_MISMATCH"):
        audit_patch_control_cohort(cohort)


def test_rejects_experiment_drift_and_duplicate_targets() -> None:
    cohort = _cohort()
    drifted = cohort["task-0"]
    cohort["task-0"] = replace(
        drifted,
        treatment=replace(drifted.treatment, sweep_spec_sha256="e" * 64).with_digest(),
        controls={
            key: replace(value, sweep_spec_sha256="e" * 64).with_digest()
            for key, value in drifted.controls.items()
        },
    )
    with pytest.raises(PatchControlCohortError, match="COHORT_EXPERIMENT_MISMATCH"):
        audit_patch_control_cohort(cohort)

    cohort = _cohort()
    duplicate = cohort["task-0"]
    original = cohort["task-1"]
    target = duplicate.treatment.target_input_sha256
    cohort["task-1"] = replace(
        original,
        treatment=replace(original.treatment, target_input_sha256=target).with_digest(),
        controls={
            key: replace(value, target_input_sha256=target).with_digest()
            for key, value in original.controls.items()
        },
    )
    with pytest.raises(PatchControlCohortError, match="COHORT_TARGET_NOT_DISTINCT"):
        audit_patch_control_cohort(cohort)


def test_cohort_inventory_and_policy_are_bounded_before_analysis() -> None:
    short = _cohort()
    short.pop("task-7")
    with pytest.raises(PatchControlCohortError, match="COHORT_TASK_COUNT_OUT_OF_BOUNDS"):
        audit_patch_control_cohort(short)

    cohort = _cohort()
    cohort["bad task id"] = cohort.pop("task-7")
    with pytest.raises(PatchControlCohortError, match="INVALID_COHORT_TASK_ID"):
        audit_patch_control_cohort(cohort)

    with pytest.raises(ValueError, match="min_tasks"):
        PatchControlCohortPolicy(min_tasks=3)
    with pytest.raises(ValueError, match="family_confidence"):
        PatchControlCohortPolicy(family_confidence=1.0)
    with pytest.raises(ValueError, match="min_tasks cannot exceed"):
        PatchControlCohortPolicy(min_tasks=9, max_tasks=8)
