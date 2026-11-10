from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from math import isfinite
from typing import Any


@dataclass(frozen=True, slots=True)
class ConsolidationPolicy:
    min_stable_layer_fraction: float = 0.25
    max_sign_flips: int = 2
    min_final_abs_margin: float = 1e-6
    epsilon: float = 1e-12
    max_layers: int = 1_000
    max_candidate_pairs: int = 496

    def __post_init__(self) -> None:
        if (
            type(self.min_stable_layer_fraction) not in {int, float}
            or not isfinite(float(self.min_stable_layer_fraction))
            or not 0 <= self.min_stable_layer_fraction <= 1
        ):
            raise ValueError("min_stable_layer_fraction must be finite and in [0, 1]")
        if type(self.max_sign_flips) is not int or self.max_sign_flips < 0:
            raise ValueError("max_sign_flips must be a non-negative integer")
        if (
            type(self.min_final_abs_margin) not in {int, float}
            or not isfinite(float(self.min_final_abs_margin))
            or self.min_final_abs_margin < 0
        ):
            raise ValueError("min_final_abs_margin must be finite and non-negative")
        if type(self.epsilon) not in {int, float} or not isfinite(float(self.epsilon)) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        for name, value in (
            ("max_layers", self.max_layers),
            ("max_candidate_pairs", self.max_candidate_pairs),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


DEFAULT_CONSOLIDATION_POLICY = ConsolidationPolicy()


@dataclass(frozen=True, slots=True)
class PairConsolidation:
    positive: str
    negative: str
    layers: int
    final_margin: float
    final_winner: str | None
    stable_from_layer: int | None
    stable_layer_fraction: float
    sign_flip_count: int
    last_flip_layer: int | None


@dataclass(frozen=True, slots=True)
class ConsolidationReport:
    decision: str
    pairs: tuple[PairConsolidation, ...]
    policy: ConsolidationPolicy
    violations: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        pair_rows = [asdict(pair) for pair in self.pairs]
        return {
            "schema_version": 1,
            "metric": "pairwise_candidate_decision_consolidation",
            "decision": self.decision,
            "summary": {
                "pairs": len(pair_rows),
                "minimum_stable_layer_fraction": min(pair.stable_layer_fraction for pair in self.pairs),
                "maximum_sign_flip_count": max(pair.sign_flip_count for pair in self.pairs),
                "minimum_final_abs_margin": min(abs(pair.final_margin) for pair in self.pairs),
            },
            "pairs": pair_rows,
            "policy": asdict(self.policy),
            "violations": list(self.violations),
        }


def _sign(value: float, *, epsilon: float) -> int:
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def _validated_candidates(candidates: Sequence[str], policy: ConsolidationPolicy) -> tuple[str, ...]:
    labels = tuple(candidates)
    if not 2 <= len(labels) <= 32:
        raise ValueError("candidate labels must contain between 2 and 32 items")
    if any(not isinstance(label, str) or not 1 <= len(label) <= 256 for label in labels):
        raise ValueError("candidate labels must be 1-256 character strings")
    if len(set(labels)) != len(labels):
        raise ValueError("candidate labels must be unique")
    pair_count = len(labels) * (len(labels) - 1) // 2
    if pair_count > policy.max_candidate_pairs:
        raise ValueError(f"candidate pairs exceed the {policy.max_candidate_pairs} pair budget")
    return labels


def _validated_layers(
    analysis: Mapping[str, Any], labels: tuple[str, ...], policy: ConsolidationPolicy
) -> tuple[list[int], dict[str, list[float]]]:
    layers = analysis.get("layers")
    if not isinstance(layers, list) or len(layers) < 2:
        raise ValueError("analysis must contain at least two layer rows")
    if len(layers) > policy.max_layers:
        raise ValueError(f"analysis exceeds the {policy.max_layers} layer budget")
    indices: list[int] = []
    scores = {label: [] for label in labels}
    for position, row in enumerate(layers, start=1):
        if not isinstance(row, Mapping):
            raise TypeError(f"layer row {position} must be a mapping")
        layer = row.get("layer")
        if type(layer) is not int or layer < 0:
            raise ValueError(f"layer row {position} has an invalid layer index")
        if indices and layer <= indices[-1]:
            raise ValueError("layer indices must be strictly increasing")
        candidate_scores = row.get("candidate_scores")
        if not isinstance(candidate_scores, Mapping):
            raise TypeError(f"layer {layer} candidate_scores must be a mapping")
        for label in labels:
            value = candidate_scores.get(label)
            if type(value) not in {int, float} or not isfinite(float(value)):
                raise ValueError(f"layer {layer} score for candidate {label!r} must be finite")
            scores[label].append(float(value))
        indices.append(layer)
    return indices, scores


def _pair_metrics(
    positive: str,
    negative: str,
    indices: list[int],
    scores: dict[str, list[float]],
    policy: ConsolidationPolicy,
) -> PairConsolidation:
    margins = [
        positive_score - negative_score
        for positive_score, negative_score in zip(scores[positive], scores[negative], strict=True)
    ]
    signs = [_sign(margin, epsilon=policy.epsilon) for margin in margins]
    previous = 0
    flip_count = 0
    last_flip_layer: int | None = None
    for layer, sign in zip(indices, signs, strict=True):
        if sign == 0:
            continue
        if previous != 0 and sign != previous:
            flip_count += 1
            last_flip_layer = layer
        previous = sign

    final_sign = signs[-1]
    stable_start = len(signs)
    if final_sign:
        stable_start = len(signs) - 1
        while stable_start > 0 and signs[stable_start - 1] == final_sign:
            stable_start -= 1
    stable_fraction = (len(signs) - stable_start) / len(signs) if final_sign else 0.0
    return PairConsolidation(
        positive=positive,
        negative=negative,
        layers=len(indices),
        final_margin=margins[-1],
        final_winner=positive if final_sign > 0 else negative if final_sign < 0 else None,
        stable_from_layer=indices[stable_start] if final_sign else None,
        stable_layer_fraction=stable_fraction,
        sign_flip_count=flip_count,
        last_flip_layer=last_flip_layer,
    )


def audit_decision_consolidation(
    analysis: Mapping[str, Any],
    candidate_labels: Sequence[str],
    *,
    policy: ConsolidationPolicy = DEFAULT_CONSOLIDATION_POLICY,
) -> ConsolidationReport:
    """Gate whether pairwise candidate preferences consolidate before the final layer."""
    if not isinstance(analysis, Mapping):
        raise TypeError("analysis must be a mapping")
    labels = _validated_candidates(candidate_labels, policy)
    indices, scores = _validated_layers(analysis, labels, policy)
    pairs = tuple(
        _pair_metrics(positive, negative, indices, scores, policy)
        for positive, negative in combinations(labels, 2)
    )
    violations: list[dict[str, Any]] = []
    for pair in pairs:
        pair_name = f"{pair.positive}::{pair.negative}"
        if abs(pair.final_margin) < policy.min_final_abs_margin:
            violations.append(
                {
                    "code": "final_margin_below_minimum",
                    "pair": pair_name,
                    "observed": abs(pair.final_margin),
                    "threshold": policy.min_final_abs_margin,
                }
            )
        if pair.stable_layer_fraction < policy.min_stable_layer_fraction:
            violations.append(
                {
                    "code": "stable_layer_fraction_below_minimum",
                    "pair": pair_name,
                    "observed": pair.stable_layer_fraction,
                    "threshold": policy.min_stable_layer_fraction,
                }
            )
        if pair.sign_flip_count > policy.max_sign_flips:
            violations.append(
                {
                    "code": "sign_flip_count_above_maximum",
                    "pair": pair_name,
                    "observed": pair.sign_flip_count,
                    "threshold": policy.max_sign_flips,
                }
            )
    violations.sort(key=lambda item: (str(item["pair"]), str(item["code"])))
    return ConsolidationReport(
        decision="pass" if not violations else "fail",
        pairs=pairs,
        policy=policy,
        violations=tuple(violations),
    )
