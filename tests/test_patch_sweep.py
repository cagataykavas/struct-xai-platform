from __future__ import annotations

import pytest

from structxai.patch_sweep import summarize_patch_sweep


def _record(layer: int, baseline_a: float, patched_a: float) -> dict:
    return {
        "layer": layer,
        "baseline_scores": {"A": baseline_a, "B": 0.0},
        "patched_scores": {"A": patched_a, "B": 0.0},
        "score_deltas": {"A": patched_a - baseline_a, "B": 0.0},
    }


def test_patch_sweep_finds_peak_layer_and_candidate_flip() -> None:
    summary = summarize_patch_sweep(
        [
            _record(0, 2.0, 1.5),
            _record(1, 2.0, -1.0),
            _record(2, 2.0, 3.0),
        ],
        "A",
        "B",
    )

    assert summary.peak_effect_layer == 1
    assert summary.peak_signed_effect == -3.0
    assert summary.mean_absolute_effect == 1.5
    assert summary.effect_concentration == pytest.approx(2 / 3)
    assert summary.direction_consistency == pytest.approx(5 / 9)
    assert summary.winner_flip_layers == (1,)
    assert sum(layer.effect_share for layer in summary.layers) == pytest.approx(1.0)


def test_patch_sweep_sorts_layers_and_serializes_json_ready_rows() -> None:
    summary = summarize_patch_sweep(
        [_record(9, 1.0, 1.5), _record(3, 1.0, 1.0)],
        "A",
        "B",
    )

    payload = summary.to_dict()
    assert [row["layer"] for row in payload["layers"]] == [3, 9]
    assert payload["winner_flip_layers"] == []
    assert payload["metric"] == "pairwise_candidate_margin_activation_patch_effect"


def test_patch_sweep_rejects_baseline_drift() -> None:
    with pytest.raises(ValueError, match="baseline candidate margin changed"):
        summarize_patch_sweep(
            [_record(0, 1.0, 2.0), _record(1, 1.1, 2.0)],
            "A",
            "B",
        )


def test_patch_sweep_rejects_duplicate_layers_and_missing_candidates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        summarize_patch_sweep(
            [_record(2, 1.0, 2.0), _record(2, 1.0, 2.0)],
            "A",
            "B",
        )

    with pytest.raises(ValueError, match="missing candidates"):
        summarize_patch_sweep(
            [{"layer": 0, "baseline_scores": {"A": 1.0}, "patched_scores": {"A": 2.0}}],
            "A",
            "B",
        )


def test_zero_effect_sweep_has_well_defined_zero_metrics() -> None:
    summary = summarize_patch_sweep(
        [_record(0, 1.0, 1.0), _record(1, 1.0, 1.0)],
        "A",
        "B",
    )

    assert summary.peak_effect_layer == 0
    assert summary.effect_concentration == 0.0
    assert summary.direction_consistency == 0.0
    assert all(layer.effect_share == 0.0 for layer in summary.layers)
