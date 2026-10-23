from dataclasses import replace

import pytest

from structxai.patch_dose_response import (
    CrossSeedDoseResponseRejected,
    CrossSeedPolicy,
    PatchSweepError,
    PatchSweepPoint,
    PatchSweepResult,
    audit_cross_seed_dose_response,
    require_cross_seed_dose_response_acceptance,
)


def sweep(
    curves: dict[int, tuple[float, ...]],
    *,
    alphas: tuple[float, ...] = (0.0, 0.5, 1.0),
) -> PatchSweepResult:
    target_margin = -1.0
    source_margin = 1.0
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
        source_input_sha256="a" * 64,
        target_input_sha256="b" * 64,
        sweep_spec_sha256="c" * 64,
        policy_sha256="d" * 64,
        source_margin=source_margin,
        target_margin=target_margin,
        repeated_target_margin=target_margin,
        model_calls=3 + len(points),
        points=points,
        evidence_sha256="",
    ).with_digest()


def stable_runs() -> dict[int, PatchSweepResult]:
    return {
        11: sweep({2: (0.0, 0.30, 0.70), 5: (0.0, 0.40, 0.85)}),
        22: sweep({2: (0.0, 0.32, 0.74), 5: (0.0, 0.42, 0.82)}),
        33: sweep({2: (0.0, 0.29, 0.68), 5: (0.0, 0.38, 0.88)}),
    }


def test_cross_seed_gate_accepts_stable_prespecified_layers() -> None:
    audit = audit_cross_seed_dose_response(
        stable_runs(),
        cross_seed_policy=CrossSeedPolicy(min_stable_layers=2),
    )

    assert audit.status == "accepted"
    assert audit.stable_layers == 2
    assert audit.seed_runs[0].seed == 11
    assert audit.layers[0].mean_endpoint_recovery == pytest.approx((0.70 + 0.74 + 0.68) / 3)
    assert audit.layers[0].endpoint_recovery_range == pytest.approx(0.06)
    assert audit.layers[0].seed_qualification_fraction == 1.0
    assert audit.layers[0].endpoint_sign_agreement == 1.0
    assert len(audit.evidence_sha256) == 64


def test_cross_seed_gate_reports_each_stability_failure() -> None:
    runs = {
        1: sweep({2: (0.0, 0.30, 0.70)}),
        2: sweep({2: (0.0, -0.20, -0.10)}),
        3: sweep({2: (0.0, 0.80, 1.20)}),
    }
    audit = audit_cross_seed_dose_response(
        runs,
        cross_seed_policy=CrossSeedPolicy(
            min_seed_qualification_fraction=0.80,
            min_endpoint_sign_agreement=0.80,
            max_endpoint_range=0.20,
            max_auc_range=0.20,
        ),
    )

    assert audit.status == "rejected"
    assert audit.finding_codes == ("STABLE_LAYER_COUNT_BELOW_MINIMUM",)
    assert audit.layers[0].finding_codes == (
        "SEED_QUALIFICATION_FRACTION_BELOW_MINIMUM",
        "ENDPOINT_SIGN_AGREEMENT_BELOW_MINIMUM",
        "ENDPOINT_RANGE_EXCEEDED",
        "AUC_RANGE_EXCEEDED",
    )


def test_cross_seed_gate_raises_stable_scientific_rejection() -> None:
    runs = {
        1: sweep({2: (0.0, 0.30, 0.70)}),
        2: sweep({2: (0.0, 0.30, -0.10)}),
        3: sweep({2: (0.0, 0.30, 1.20)}),
    }

    with pytest.raises(CrossSeedDoseResponseRejected, match="CROSS_SEED_DOSE_RESPONSE_REJECTED"):
        require_cross_seed_dose_response_acceptance(
            runs,
            cross_seed_policy=CrossSeedPolicy(max_endpoint_range=0.10),
        )


def test_cross_seed_gate_rejects_mismatched_experiment_identity() -> None:
    runs = stable_runs()
    runs[33] = replace(runs[33], sweep_spec_sha256="e" * 64).with_digest()

    with pytest.raises(PatchSweepError, match="CROSS_SEED_EXPERIMENT_MISMATCH"):
        audit_cross_seed_dose_response(runs)


def test_cross_seed_gate_revalidates_each_child_digest() -> None:
    runs = stable_runs()
    runs[22] = replace(runs[22], target_margin=-2.0)

    with pytest.raises(PatchSweepError, match="SWEEP_EVIDENCE_DIGEST_MISMATCH"):
        audit_cross_seed_dose_response(runs)


@pytest.mark.parametrize(
    ("runs", "code"),
    [
        ({1: sweep({2: (0.0, 0.30, 0.70)})}, "SEED_RUN_COUNT_OUT_OF_BOUNDS"),
        (
            {
                -1: sweep({2: (0.0, 0.30, 0.70)}),
                2: sweep({2: (0.0, 0.31, 0.71)}),
                3: sweep({2: (0.0, 0.32, 0.72)}),
            },
            "INVALID_SEED_ID",
        ),
    ],
)
def test_cross_seed_gate_bounds_and_validates_seed_set(runs: dict[int, PatchSweepResult], code: str) -> None:
    with pytest.raises(PatchSweepError, match=code):
        audit_cross_seed_dose_response(runs)


def test_cross_seed_gate_rejects_mixed_seed_types_with_stable_error() -> None:
    valid = sweep({2: (0.0, 0.30, 0.70)})

    with pytest.raises(PatchSweepError, match="INVALID_SEED_ID"):
        audit_cross_seed_dose_response({1: valid, "2": valid, 3: valid})  # type: ignore[dict-item]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"min_seed_runs": 1}, "min_seed_runs"),
        ({"max_seed_runs": 1}, "max_seed_runs"),
        ({"min_seed_runs": 4, "max_seed_runs": 3}, "cannot exceed"),
        ({"min_seed_qualification_fraction": 1.1}, "min_seed_qualification_fraction"),
        ({"min_endpoint_sign_agreement": float("nan")}, "min_endpoint_sign_agreement"),
        ({"max_endpoint_range": -0.1}, "max_endpoint_range"),
        ({"max_auc_range": float("inf")}, "max_auc_range"),
        ({"sign_atol": -1.0}, "sign_atol"),
        ({"min_stable_layers": 0}, "min_stable_layers"),
    ],
)
def test_cross_seed_policy_is_bounded(overrides: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        CrossSeedPolicy(**overrides)


def test_cross_seed_evidence_is_order_independent_and_binds_seed_labels_and_policy() -> None:
    runs = stable_runs()
    reordered = {33: runs[33], 11: runs[11], 22: runs[22]}
    relabelled = {11: runs[11], 22: runs[22], 44: runs[33]}

    first = audit_cross_seed_dose_response(runs)
    second = audit_cross_seed_dose_response(reordered)
    changed_seed = audit_cross_seed_dose_response(relabelled)
    stricter = audit_cross_seed_dose_response(
        runs,
        cross_seed_policy=CrossSeedPolicy(max_endpoint_range=0.05),
    )

    assert first.as_dict() == second.as_dict()
    assert first.evidence_sha256 != changed_seed.evidence_sha256
    assert first.cross_seed_policy_sha256 != stricter.cross_seed_policy_sha256
    assert first.evidence_sha256 != stricter.evidence_sha256
