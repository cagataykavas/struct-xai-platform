from __future__ import annotations

from itertools import combinations
from statistics import mean


def _margin_trace(analysis: dict, positive: str, negative: str) -> list[float]:
    return [
        float(layer["candidate_scores"][positive])
        - float(layer["candidate_scores"][negative])
        for layer in analysis["layers"]
    ]


def _sign_flip_count(values: list[float]) -> int:
    flips = 0
    previous = 0.0
    for value in values:
        if value == 0:
            continue
        if previous != 0 and (previous > 0) != (value > 0):
            flips += 1
        previous = value
    return flips


def _trace_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("margin trace cannot be empty")
    return {
        "layers": len(values),
        "initial_margin": values[0],
        "final_margin": values[-1],
        "mean_abs_margin": mean(abs(value) for value in values),
        "max_abs_margin": max(abs(value) for value in values),
        "sign_flip_count": _sign_flip_count(values),
    }


def _intervention_summary(
    base: dict,
    variant: dict,
    positive: str,
    negative: str,
) -> dict[str, object]:
    base_trace = _margin_trace(base, positive, negative)
    variant_trace = _margin_trace(variant, positive, negative)
    if len(base_trace) != len(variant_trace):
        raise ValueError("base and intervention analyses must contain the same number of layers")

    deltas = [variant_value - base_value for base_value, variant_value in zip(base_trace, variant_trace, strict=True)]
    peak_index = max(range(len(deltas)), key=lambda index: abs(deltas[index]))
    final_base = base_trace[-1]
    final_variant = variant_trace[-1]

    return {
        "base_final_margin": final_base,
        "variant_final_margin": final_variant,
        "final_margin_delta": final_variant - final_base,
        "mean_abs_margin_delta": mean(abs(value) for value in deltas),
        "max_abs_margin_delta": abs(deltas[peak_index]),
        "peak_effect_layer": int(base["layers"][peak_index]["layer"]),
        "pairwise_winner_changed_at_final_layer": (
            final_base != 0
            and final_variant != 0
            and (final_base > 0) != (final_variant > 0)
        ),
        "variant_sign_flip_count": _sign_flip_count(variant_trace),
    }


def evaluate_interventions(
    base: dict,
    variants: dict[str, dict],
    candidate_labels: tuple[str, ...] | list[str],
) -> dict[str, object]:
    """Quantify how interventions move candidate margins across layers.

    This evaluates the *existing first-token candidate logit margin* used by the
    project. It does not claim to measure full-sequence causal attribution. For
    every candidate pair it reports the baseline margin trace and how strongly
    each deletion/replacement changes that trace.
    """
    labels = tuple(candidate_labels)
    if len(labels) < 2:
        raise ValueError("at least two candidates are required for evaluation")

    pair_rows: list[dict[str, object]] = []
    for positive, negative in combinations(labels, 2):
        base_trace = _margin_trace(base, positive, negative)
        intervention_rows = {
            name: _intervention_summary(
                base,
                payload["analysis"],
                positive,
                negative,
            )
            for name, payload in variants.items()
        }
        pair_rows.append(
            {
                "positive": positive,
                "negative": negative,
                "baseline": _trace_summary(base_trace),
                "interventions": intervention_rows,
            }
        )

    return {
        "metric": "pairwise_first_token_candidate_logit_margin_shift",
        "interpretation": (
            "larger absolute margin shifts indicate stronger intervention sensitivity; "
            "they are not full-sequence attribution scores"
        ),
        "pairs": pair_rows,
    }
