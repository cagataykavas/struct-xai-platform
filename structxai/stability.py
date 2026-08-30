from __future__ import annotations

from itertools import combinations
from math import sqrt
from statistics import mean


def _margin_trace(analysis: dict, positive: str, negative: str) -> list[float]:
    layers = analysis.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("analysis must contain a non-empty layers list")
    trace: list[float] = []
    for row in layers:
        scores = row["candidate_scores"]
        trace.append(float(scores[positive]) - float(scores[negative]))
    return trace


def _layer_indices(analysis: dict) -> list[int]:
    return [int(row["layer"]) for row in analysis["layers"]]


def _sign(value: float, *, eps: float = 1e-12) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("traces must have equal length")
    if not left:
        raise ValueError("traces cannot be empty")
    if len(left) == 1:
        return 1.0 if left[0] == right[0] else 0.0

    left_mean = mean(left)
    right_mean = mean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    numerator = sum(a * b for a, b in zip(left_centered, right_centered, strict=True))
    left_norm = sqrt(sum(value * value for value in left_centered))
    right_norm = sqrt(sum(value * value for value in right_centered))

    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0 if left == right else 0.0
    return numerator / (left_norm * right_norm)


def compare_runs(
    left: dict,
    right: dict,
    candidate_labels: tuple[str, ...] | list[str],
) -> dict[str, object]:
    """Compare two layer-wise explanation runs for the same experiment.

    Stability is evaluated on the project's explicit pairwise first-token
    candidate-logit margins. The result does not claim that a high correlation
    proves causal faithfulness; it only quantifies repeatability of the measured
    decision trace.
    """
    labels = tuple(candidate_labels)
    if len(labels) < 2:
        raise ValueError("at least two candidates are required")
    if _layer_indices(left) != _layer_indices(right):
        raise ValueError("runs must contain the same layer indices")

    pairs: list[dict[str, object]] = []
    for positive, negative in combinations(labels, 2):
        left_trace = _margin_trace(left, positive, negative)
        right_trace = _margin_trace(right, positive, negative)
        deltas = [
            right_value - left_value
            for left_value, right_value in zip(left_trace, right_trace, strict=True)
        ]
        sign_matches = [
            _sign(left_value) == _sign(right_value)
            for left_value, right_value in zip(left_trace, right_trace, strict=True)
        ]
        pairs.append(
            {
                "positive": positive,
                "negative": negative,
                "pearson_correlation": _pearson(left_trace, right_trace),
                "mean_abs_margin_difference": mean(abs(value) for value in deltas),
                "max_abs_margin_difference": max(abs(value) for value in deltas),
                "sign_agreement_rate": sum(sign_matches) / len(sign_matches),
                "final_margin_difference": right_trace[-1] - left_trace[-1],
                "final_pairwise_winner_agreement": _sign(left_trace[-1]) == _sign(right_trace[-1]),
                "layers": len(left_trace),
            }
        )

    return {
        "metric": "cross_run_pairwise_margin_stability",
        "interpretation": (
            "higher correlation/sign agreement and lower absolute margin differences "
            "indicate more repeatable layer-wise decision traces"
        ),
        "pairs": pairs,
        "summary": {
            "mean_pearson_correlation": mean(float(row["pearson_correlation"]) for row in pairs),
            "mean_sign_agreement_rate": mean(float(row["sign_agreement_rate"]) for row in pairs),
            "mean_abs_margin_difference": mean(
                float(row["mean_abs_margin_difference"]) for row in pairs
            ),
            "all_final_pairwise_winners_agree": all(
                bool(row["final_pairwise_winner_agreement"]) for row in pairs
            ),
        },
    }


def compare_artifacts(left: dict, right: dict) -> dict[str, object]:
    left_experiment = left.get("experiment")
    right_experiment = right.get("experiment")
    if not isinstance(left_experiment, dict) or not isinstance(right_experiment, dict):
        raise ValueError("both artifacts must contain experiment metadata")

    identity_fields = ("prompt", "candidates", "model_name")
    mismatches = [
        field
        for field in identity_fields
        if left_experiment.get(field) != right_experiment.get(field)
    ]
    if mismatches:
        raise ValueError(f"artifacts describe different experiments: {', '.join(mismatches)}")

    left_provenance = left.get("provenance", {})
    right_provenance = right.get("provenance", {})
    left_fingerprint = left_provenance.get("experiment_fingerprint")
    right_fingerprint = right_provenance.get("experiment_fingerprint")
    if left_fingerprint and right_fingerprint and left_fingerprint != right_fingerprint:
        raise ValueError("experiment fingerprints differ")

    candidates = tuple(str(item) for item in left_experiment["candidates"])
    result = compare_runs(left["base"], right["base"], candidates)
    result["left_created_at"] = left.get("created_at")
    result["right_created_at"] = right.get("created_at")
    result["experiment_fingerprint"] = left_fingerprint or right_fingerprint
    return result
