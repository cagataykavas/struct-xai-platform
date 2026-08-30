import pytest

from structxai.stability import compare_artifacts, compare_runs


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
