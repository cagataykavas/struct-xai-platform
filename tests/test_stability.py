import json

import pytest

from structxai.stability import (
    StabilityCohortPolicy,
    StabilityGateError,
    audit_artifact_stability_cohort,
    compare_artifacts,
    compare_runs,
    require_artifact_stability_cohort,
)


def _analysis(margins: list[float]) -> dict:
    return {
        "layers": [
            {
                "layer": index,
                "candidate_scores": {"A": margin, "B": 0.0},
            }
            for index, margin in enumerate(margins)
        ]
    }


def _artifact(margins: list[float], *, prompt: str = "Question?") -> dict:
    return {
        "experiment": {
            "name": "repeatability-demo",
            "prompt": prompt,
            "candidates": ["A", "B"],
            "model_name": "tiny-model",
        },
        "created_at": "2026-08-30T10:00:00+00:00",
        "base": _analysis(margins),
        "provenance": {},
    }


def _cohort_artifact(
    margins: list[float],
    *,
    prompt: str = "Private cohort prompt",
    fingerprint: str = "a" * 64,
) -> dict:
    artifact = _artifact(margins, prompt=prompt)
    artifact["provenance"] = {"experiment_fingerprint": fingerprint}
    return artifact


def test_identical_runs_are_perfectly_stable():
    result = compare_runs(_analysis([1.0, 2.0, -1.0]), _analysis([1.0, 2.0, -1.0]), ["A", "B"])
    pair = result["pairs"][0]
    assert pair["pearson_correlation"] == pytest.approx(1.0)
    assert pair["mean_abs_margin_difference"] == pytest.approx(0.0)
    assert pair["sign_agreement_rate"] == pytest.approx(1.0)
    assert pair["final_pairwise_winner_agreement"] is True
    assert result["summary"]["all_final_pairwise_winners_agree"] is True


def test_stability_detects_trace_drift_and_final_winner_change():
    result = compare_runs(_analysis([1.0, 2.0, 1.0]), _analysis([1.1, 1.4, -0.5]), ["A", "B"])
    pair = result["pairs"][0]
    assert pair["mean_abs_margin_difference"] > 0.0
    assert pair["max_abs_margin_difference"] >= pair["mean_abs_margin_difference"]
    assert pair["sign_agreement_rate"] < 1.0
    assert pair["final_pairwise_winner_agreement"] is False


def test_artifact_comparison_rejects_different_prompt():
    with pytest.raises(ValueError, match="different experiments"):
        compare_artifacts(
            _artifact([1.0, 2.0], prompt="Prompt A"),
            _artifact([1.0, 2.0], prompt="Prompt B"),
        )


def test_artifact_comparison_reports_run_metadata():
    left = _artifact([1.0, 2.0])
    right = _artifact([1.2, 2.1])
    right["created_at"] = "2026-08-30T10:05:00+00:00"
    result = compare_artifacts(left, right)
    assert result["left_created_at"] == "2026-08-30T10:00:00+00:00"
    assert result["right_created_at"] == "2026-08-30T10:05:00+00:00"


def test_stability_cohort_accepts_repeatable_runs_with_deterministic_evidence():
    runs = {
        "seed-33": _cohort_artifact([1.02, 2.01, 3.01]),
        "seed-11": _cohort_artifact([1.00, 2.00, 3.00]),
        "seed-22": _cohort_artifact([1.01, 1.99, 3.02]),
    }

    first = audit_artifact_stability_cohort(runs)
    second = audit_artifact_stability_cohort(dict(reversed(tuple(runs.items()))))

    assert first == second
    assert first["accepted"] is True
    assert first["run_count"] == 3
    assert first["pairwise_comparison_count"] == 3
    assert first["finding_codes"] == []
    assert len(first["evidence_sha256"]) == 64
    assert "Private cohort prompt" not in json.dumps(first)
    assert require_artifact_stability_cohort(runs) == first


def test_stability_cohort_reports_prespecified_failure_codes():
    runs = {
        "seed-11": _cohort_artifact([1.0, 2.0, 3.0]),
        "seed-22": _cohort_artifact([1.0, 2.0, 3.0]),
        "seed-33": _cohort_artifact([1.0, 2.0, -1.0]),
    }

    report = audit_artifact_stability_cohort(runs)

    assert report["accepted"] is False
    assert set(report["finding_codes"]) == {
        "FINAL_MARGIN_RANGE_ABOVE_LIMIT",
        "FINAL_WINNER_DISAGREEMENT",
        "PAIRWISE_MARGIN_DRIFT_ABOVE_LIMIT",
        "PAIRWISE_PEARSON_BELOW_FLOOR",
        "PAIRWISE_SIGN_AGREEMENT_BELOW_FLOOR",
    }
    with pytest.raises(StabilityGateError, match="policy failed"):
        require_artifact_stability_cohort(runs)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda artifact: artifact["provenance"].update(experiment_fingerprint="b" * 64), "fingerprints"),
        (lambda artifact: artifact["experiment"].update(prompt="Different"), "different experiments"),
        (lambda artifact: artifact["base"]["layers"].pop(), "different layer indices"),
        (
            lambda artifact: artifact["base"]["layers"][0]["candidate_scores"].update(A=float("nan")),
            "finite numbers",
        ),
    ],
)
def test_stability_cohort_rejects_incomparable_or_unsafe_runs(mutate, match):
    runs = {
        "seed-11": _cohort_artifact([1.0, 2.0, 3.0]),
        "seed-22": _cohort_artifact([1.0, 2.0, 3.0]),
        "seed-33": _cohort_artifact([1.0, 2.0, 3.0]),
    }
    mutate(runs["seed-33"])

    with pytest.raises(StabilityGateError, match=match):
        audit_artifact_stability_cohort(runs)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"min_runs": 2}, "min_runs"),
        ({"min_runs": 4, "max_runs": 3}, "max_runs"),
        ({"min_pairwise_pearson": float("nan")}, "finite"),
        ({"min_pairwise_sign_agreement": 1.1}, "between zero and one"),
        ({"max_final_margin_range": -0.1}, "non-negative"),
        ({"require_final_winner_unanimity": 1}, "boolean"),
    ],
)
def test_stability_cohort_policy_is_bounded(kwargs, match):
    with pytest.raises(ValueError, match=match):
        StabilityCohortPolicy(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_pairwise_pearson": True},
        {"max_pairwise_mean_abs_margin_difference": "0.05"},
    ],
)
def test_stability_cohort_policy_rejects_non_numeric_thresholds(kwargs):
    with pytest.raises(TypeError, match="must be numeric"):
        StabilityCohortPolicy(**kwargs)


def test_stability_cohort_requires_enough_runs_and_canonical_ids():
    with pytest.raises(StabilityGateError, match="run count"):
        audit_artifact_stability_cohort(
            {
                "seed-11": _cohort_artifact([1.0, 2.0]),
                "seed-22": _cohort_artifact([1.0, 2.0]),
            }
        )
    with pytest.raises(StabilityGateError, match="run IDs"):
        audit_artifact_stability_cohort(
            {
                "seed-11": _cohort_artifact([1.0, 2.0]),
                "seed-22": _cohort_artifact([1.0, 2.0]),
                "unsafe id": _cohort_artifact([1.0, 2.0]),
            }
        )
