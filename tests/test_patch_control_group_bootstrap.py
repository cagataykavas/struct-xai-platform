from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from structxai.patch_control_cohort import PatchControlCohortPolicy, PatchControlTask
from structxai.patch_control_group_bootstrap import (
    GROUP_BOOTSTRAP_METHOD,
    PatchControlGroupBootstrapError,
    PatchControlGroupBootstrapPolicy,
    PatchControlGroupBootstrapRejected,
    audit_patch_control_group_bootstrap,
    require_patch_control_group_bootstrap,
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
        model_revision="sha256:group-bootstrap-model",
        source_input_sha256=_digest(source),
        target_input_sha256=_digest(target),
        sweep_spec_sha256="a" * 64,
        policy_sha256="b" * 64,
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


def _cohort(size: int = 8, *, failing: set[int] | None = None) -> dict[str, PatchControlTask]:
    failures = failing or set()
    return {f"task-{index}": _task(index, specific=index not in failures) for index in range(size)}


def _paired_groups(size: int = 8) -> dict[str, str]:
    return {f"task-{index}": f"template-{index // 2}" for index in range(size)}


def test_accepts_prespecified_groups_with_familywise_bootstrap_bounds() -> None:
    audit = require_patch_control_group_bootstrap(_cohort(), _paired_groups())

    assert audit.status == "accepted"
    assert audit.method == GROUP_BOOTSTRAP_METHOD
    assert audit.resampling_unit == "template_group"
    assert audit.hypothesis_count == 2
    assert audit.per_hypothesis_alpha == pytest.approx(0.025)
    assert audit.bootstrap_work_units == 16_000
    assert audit.group_specific_layers == 2
    assert all(layer.bootstrap_lower_bound == 1.0 for layer in audit.layers)
    assert len(audit.evidence_sha256) == 64


def test_large_related_family_cannot_pad_independent_evidence() -> None:
    tasks = _cohort(20, failing=set(range(14, 20)))
    groups = {
        **{f"task-{index}": "large-success-family" for index in range(14)},
        **{f"task-{index}": f"failed-family-{(index - 14) // 2}" for index in range(14, 20)},
    }
    base_policy = PatchControlCohortPolicy(min_specific_task_fraction_lower_bound=0.0)

    audit = audit_patch_control_group_bootstrap(tasks, groups, cohort_policy=base_policy)

    assert all(layer.observed_task_fraction == pytest.approx(0.70) for layer in audit.layers)
    assert all(layer.observed_group_mean_fraction == pytest.approx(0.25) for layer in audit.layers)
    assert all(layer.bootstrap_lower_bound < 0.50 for layer in audit.layers)
    assert audit.status == "rejected"
    assert audit.finding_codes == ("GROUP_SPECIFIC_LAYER_COUNT_BELOW_MINIMUM",)
    with pytest.raises(
        PatchControlGroupBootstrapRejected,
        match="PATCH_CONTROL_GROUP_BOOTSTRAP_REJECTED",
    ):
        require_patch_control_group_bootstrap(tasks, groups, cohort_policy=base_policy)


def test_grouped_receipt_is_order_independent_and_omits_raw_identifiers() -> None:
    tasks = _cohort()
    groups = _paired_groups()

    first = audit_patch_control_group_bootstrap(tasks, groups)
    second = audit_patch_control_group_bootstrap(
        dict(reversed(tuple(tasks.items()))),
        dict(reversed(tuple(groups.items()))),
    )
    serialized = json.dumps(first.as_dict(), sort_keys=True)

    assert first == second
    assert "task-0" not in serialized
    assert "template-0" not in serialized
    assert "null-0" not in serialized
    assert all(len(task.task_id_sha256) == 64 for task in first.tasks)
    assert all(len(group.group_id_sha256) == 64 for group in first.groups)


def test_base_cohort_rejection_is_retained_by_group_gate() -> None:
    tasks = _cohort(failing={0, 1, 2, 3})
    audit = audit_patch_control_group_bootstrap(tasks, _paired_groups())

    assert audit.status == "rejected"
    assert "BASE_TASK_COHORT_REJECTED" in audit.finding_codes


def test_child_sweeps_are_revalidated_before_resampling() -> None:
    tasks = _cohort()
    task = tasks["task-0"]
    tasks["task-0"] = replace(
        task,
        treatment=replace(task.treatment, evidence_sha256="0" * 64),
    )

    with pytest.raises(PatchSweepError, match="SWEEP_EVIDENCE_DIGEST_MISMATCH"):
        audit_patch_control_group_bootstrap(tasks, _paired_groups())


def test_group_inventory_and_cardinality_fail_closed() -> None:
    tasks = _cohort()
    missing = _paired_groups()
    missing.pop("task-7")
    with pytest.raises(PatchControlGroupBootstrapError, match="TASK_GROUP_INVENTORY_MISMATCH"):
        audit_patch_control_group_bootstrap(tasks, missing)

    invalid = _paired_groups()
    invalid["task-7"] = "unsafe group"
    with pytest.raises(PatchControlGroupBootstrapError, match="INVALID_TASK_GROUP_ID"):
        audit_patch_control_group_bootstrap(tasks, invalid)

    too_few = {task_id: f"template-{index % 3}" for index, task_id in enumerate(tasks)}
    with pytest.raises(PatchControlGroupBootstrapError, match="TASK_GROUP_COUNT_OUT_OF_BOUNDS"):
        audit_patch_control_group_bootstrap(tasks, too_few)

    concentrated = _paired_groups()
    with pytest.raises(PatchControlGroupBootstrapError, match="TASKS_PER_GROUP_OUT_OF_BOUNDS"):
        audit_patch_control_group_bootstrap(
            tasks,
            concentrated,
            group_policy=PatchControlGroupBootstrapPolicy(max_tasks_per_group=1),
        )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"min_groups": 3}, "min_groups"),
        ({"min_groups": 5, "max_groups": 4}, "min_groups cannot exceed"),
        ({"bootstrap_resamples": 499}, "bootstrap_resamples"),
        ({"bootstrap_seed": -1}, "bootstrap_seed"),
        ({"family_confidence": float("nan")}, "finite"),
        ({"min_group_specific_fraction_lower_bound": 1.1}, "between"),
    ],
)
def test_group_bootstrap_policy_is_bounded(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        PatchControlGroupBootstrapPolicy(**kwargs)


def test_group_policy_type_is_explicit() -> None:
    with pytest.raises(TypeError, match="group_policy"):
        audit_patch_control_group_bootstrap(_cohort(), _paired_groups(), group_policy=object())  # type: ignore[arg-type]
