from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isclose
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class LayerPatchEffect:
    layer: int
    baseline_margin: float
    patched_margin: float
    signed_effect: float
    absolute_effect: float
    effect_share: float
    winner_changed: bool


@dataclass(frozen=True)
class PatchSweepSummary:
    metric: str
    positive_candidate: str
    negative_candidate: str
    baseline_margin: float
    peak_effect_layer: int
    peak_signed_effect: float
    peak_absolute_effect: float
    mean_absolute_effect: float
    effect_concentration: float
    direction_consistency: float
    winner_flip_layers: tuple[int, ...]
    layers: tuple[LayerPatchEffect, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["winner_flip_layers"] = list(self.winner_flip_layers)
        payload["layers"] = [asdict(layer) for layer in self.layers]
        return payload


def _scores(record: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = record.get(field)
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _margin(scores: Mapping[str, Any], positive: str, negative: str, field: str) -> float:
    missing = [label for label in (positive, negative) if label not in scores]
    if missing:
        raise ValueError(f"{field} missing candidates: {', '.join(missing)}")
    try:
        return float(scores[positive]) - float(scores[negative])
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} candidate scores must be numeric") from exc


def summarize_patch_sweep(
    records: Sequence[Mapping[str, Any]],
    positive_candidate: str,
    negative_candidate: str,
    *,
    baseline_tolerance: float = 1e-6,
) -> PatchSweepSummary:
    """Summarize candidate-margin effects across an activation-patching layer sweep.

    Records use the JSON shape produced by ``serialize_patch``. The evaluator
    deliberately stays model-free so saved GPU experiment outputs can be
    validated, compared and reported in CPU-only CI.
    """
    if not records:
        raise ValueError("patch sweep cannot be empty")
    if not positive_candidate or not negative_candidate:
        raise ValueError("candidate labels must be non-empty")
    if positive_candidate == negative_candidate:
        raise ValueError("positive and negative candidates must differ")
    if baseline_tolerance < 0:
        raise ValueError("baseline_tolerance cannot be negative")

    parsed: list[tuple[int, float, float]] = []
    observed_layers: set[int] = set()
    for record in records:
        raw_layer = record.get("layer")
        if isinstance(raw_layer, bool) or not isinstance(raw_layer, int):
            raise TypeError("layer must be an integer")
        if raw_layer in observed_layers:
            raise ValueError(f"duplicate patch result for layer {raw_layer}")
        observed_layers.add(raw_layer)

        baseline = _margin(
            _scores(record, "baseline_scores"),
            positive_candidate,
            negative_candidate,
            "baseline_scores",
        )
        patched = _margin(
            _scores(record, "patched_scores"),
            positive_candidate,
            negative_candidate,
            "patched_scores",
        )
        parsed.append((raw_layer, baseline, patched))

    parsed.sort(key=lambda item: item[0])
    reference_baseline = parsed[0][1]
    if any(
        not isclose(baseline, reference_baseline, rel_tol=0.0, abs_tol=baseline_tolerance)
        for _, baseline, _ in parsed[1:]
    ):
        raise ValueError("baseline candidate margin changed across patch layers")

    effects = [patched - baseline for _, baseline, patched in parsed]
    total_absolute_effect = sum(abs(effect) for effect in effects)
    peak_index = max(range(len(parsed)), key=lambda index: abs(effects[index]))
    layer_rows = tuple(
        LayerPatchEffect(
            layer=layer,
            baseline_margin=baseline,
            patched_margin=patched,
            signed_effect=effect,
            absolute_effect=abs(effect),
            effect_share=(abs(effect) / total_absolute_effect if total_absolute_effect else 0.0),
            winner_changed=(
                baseline != 0.0
                and patched != 0.0
                and (baseline > 0.0) != (patched > 0.0)
            ),
        )
        for (layer, baseline, patched), effect in zip(parsed, effects, strict=True)
    )

    return PatchSweepSummary(
        metric="pairwise_candidate_margin_activation_patch_effect",
        positive_candidate=positive_candidate,
        negative_candidate=negative_candidate,
        baseline_margin=reference_baseline,
        peak_effect_layer=parsed[peak_index][0],
        peak_signed_effect=effects[peak_index],
        peak_absolute_effect=abs(effects[peak_index]),
        mean_absolute_effect=mean(abs(effect) for effect in effects),
        effect_concentration=(
            abs(effects[peak_index]) / total_absolute_effect if total_absolute_effect else 0.0
        ),
        direction_consistency=(
            abs(sum(effects)) / total_absolute_effect if total_absolute_effect else 0.0
        ),
        winner_flip_layers=tuple(row.layer for row in layer_rows if row.winner_changed),
        layers=layer_rows,
    )
