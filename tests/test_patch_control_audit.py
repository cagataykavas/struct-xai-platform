from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from structxai.patch_control_audit import (
    PatchControlAuditError,
    PatchControlAuditRejected,
    PatchControlPolicy,
    audit_patch_control_specificity,
    require_patch_control_specificity,
)
from structxai.patch_dose_response import PatchSweepError, PatchSweepPoint, PatchSweepResult


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _sweep(
    source: str,
    curves: dict[int, tuple[float, ...]],
    *,
    target_sha256: str = "b" * 64,
    spec_sha256: str = "c" * 64,
) -> PatchSweepResult:
    target_margin = -1.0
    source_margin = 1.0
    alphas = (0.0, 0.5, 1.0)
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
        for alpha, recovery in zip(alphas, recoveries, strict=True)
    )
    return PatchSweepResult(
        schema_version=1,
        model_revision="sha256:test-model",
        source_input_sha256=_digest(source),
        target_input_sha256=target_sha256,
        sweep_spec_sha256=spec_sha256,
        policy_sha256="d" * 64,
        source_margin=source_margin,
        target_margin=target_margin,
        repeated_target_margin=target_margin,
        model_calls=3 + len(points),
        points=points,
        evidence_sha256="",
    ).with_digest()


def _treatment() -> PatchSweepResult:
    return _sweep("treatment", {0: (0.0, 0.5, 1.0), 2: (0.0, 0.35, 0.7)})


def _controls() -> dict[str, PatchSweepResult]:
    return {
        "null-a": _sweep("null-a-source", {0: (0.0, 0.03, 0.06), 2: (0.0, 0.02, 0.04)}),
        "null-b": _sweep("null-b-source", {0: (0.0, 0.05, 0.10), 2: (0.0, 0.04, 0.08)}),
        "null-c": _sweep("null-c-source", {0: (0.0, -0.01, -0.02), 2: (0.0, 0.01, 0.02)}),
    }


def test_accepts_effect_that_exceeds_every_matched_control() -> None:
    audit = require_patch_control_specificity(_treatment(), _controls())

    assert audit.status == "accepted"
    assert audit.total_layers == 2
    assert audit.specific_layers == 2
    assert audit.finding_codes == ()
    assert all(layer.specific for layer in audit.layers)
    assert len(audit.evidence_sha256) == 64


def test_evidence_is_deterministic_order_independent_and_privacy_reduced() -> None:
    controls = _controls()
    first = audit_patch_control_specificity(_treatment(), controls)
    second = audit_patch_control_specificity(_treatment(), dict(reversed(tuple(controls.items()))))

    assert first == second
    serialized = json.dumps(first.as_dict(), sort_keys=True)
    assert "null-a" not in serialized
    assert all(len(item.control_id_sha256) == 64 for item in first.controls)


def test_strong_null_effect_rejects_specificity_claim() -> None:
    controls = _controls()
    controls["null-b"] = _sweep("null-b-source", {0: (0.0, 0.45, 0.90), 2: (0.0, 0.30, 0.60)})

    audit = audit_patch_control_specificity(_treatment(), controls)

    assert audit.status == "rejected"
    assert audit.specific_layers == 0
    assert audit.finding_codes == ("SPECIFIC_LAYER_COUNT_BELOW_MINIMUM",)
    assert "CONTROL_ENDPOINT_CEILING_EXCEEDED" in audit.layers[0].finding_codes
    assert "ENDPOINT_EXCESS_BELOW_MINIMUM" in audit.layers[0].finding_codes
    with pytest.raises(PatchControlAuditRejected, match="PATCH_CONTROL_SPECIFICITY_REJECTED"):
        require_patch_control_specificity(_treatment(), controls)


@pytest.mark.parametrize(
    "changed",
    [
        _sweep("other", {0: (0.0, 0.03, 0.06), 2: (0.0, 0.02, 0.04)}, target_sha256="e" * 64),
        _sweep("other", {0: (0.0, 0.03, 0.06), 2: (0.0, 0.02, 0.04)}, spec_sha256="f" * 64),
        _sweep("other", {1: (0.0, 0.03, 0.06), 2: (0.0, 0.02, 0.04)}),
    ],
)
def test_rejects_controls_from_a_different_experiment(changed: PatchSweepResult) -> None:
    controls = _controls()
    controls["null-a"] = changed

    with pytest.raises(PatchControlAuditError, match="CONTROL_EXPERIMENT_MISMATCH"):
        audit_patch_control_specificity(_treatment(), controls)


def test_rejects_reused_control_sources_and_tampered_sweeps() -> None:
    controls = _controls()
    controls["null-c"] = controls["null-b"]
    with pytest.raises(PatchControlAuditError, match="CONTROL_SOURCE_NOT_DISTINCT"):
        audit_patch_control_specificity(_treatment(), controls)

    controls = _controls()
    controls["null-a"] = replace(
        controls["null-a"],
        points=(replace(controls["null-a"].points[0], normalized_recovery=0.1),)
        + controls["null-a"].points[1:],
    )
    with pytest.raises(PatchSweepError, match="SWEEP_EVIDENCE_DIGEST_MISMATCH"):
        audit_patch_control_specificity(_treatment(), controls)


def test_control_count_and_policy_are_bounded_before_analysis() -> None:
    controls = _controls()
    controls.pop("null-c")
    with pytest.raises(PatchControlAuditError, match="CONTROL_COUNT_OUT_OF_BOUNDS"):
        audit_patch_control_specificity(_treatment(), controls)

    with pytest.raises(ValueError, match="min_controls"):
        PatchControlPolicy(min_controls=1)
    with pytest.raises(ValueError, match="min_controls cannot exceed"):
        PatchControlPolicy(min_controls=5, max_controls=4)
    with pytest.raises(ValueError, match="finite"):
        PatchControlPolicy(min_endpoint_excess=float("nan"))


def test_policy_can_require_multiple_specific_layers() -> None:
    controls = _controls()
    controls["null-b"] = _sweep("null-b-source", {0: (0.0, 0.05, 0.10), 2: (0.0, 0.29, 0.58)})
    policy = PatchControlPolicy(min_specific_layers=2)

    audit = audit_patch_control_specificity(_treatment(), controls, control_policy=policy)

    assert audit.specific_layers == 1
    assert audit.status == "rejected"
    assert audit.finding_codes == ("SPECIFIC_LAYER_COUNT_BELOW_MINIMUM",)
