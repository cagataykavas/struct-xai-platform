from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import combinations
from math import sqrt
from statistics import mean

STABILITY_COHORT_SCHEMA = "struct-xai-stability-cohort/v1"
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_MAX_LAYERS = 4_096


class StabilityGateError(ValueError):
    """A stability cohort is malformed or fails its prespecified policy."""

    code = "STABILITY_COHORT_REJECTED"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


def _canonical_digest(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise StabilityGateError("evidence is not canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class StabilityCohortPolicy:
    min_runs: int = 3
    max_runs: int = 32
    min_pairwise_pearson: float = 0.95
    min_pairwise_sign_agreement: float = 0.95
    max_pairwise_mean_abs_margin_difference: float = 0.05
    max_final_margin_range: float = 0.05
    require_final_winner_unanimity: bool = True

    def __post_init__(self) -> None:
        if type(self.min_runs) is not int or not 3 <= self.min_runs <= 64:
            raise ValueError("min_runs must be an integer from 3 through 64")
        if type(self.max_runs) is not int or not self.min_runs <= self.max_runs <= 64:
            raise ValueError("max_runs must be an integer from min_runs through 64")
        for name, value in (
            ("min_pairwise_pearson", self.min_pairwise_pearson),
            ("min_pairwise_sign_agreement", self.min_pairwise_sign_agreement),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and between zero and one")
        for name, value in (
            (
                "max_pairwise_mean_abs_margin_difference",
                self.max_pairwise_mean_abs_margin_difference,
            ),
            ("max_final_margin_range", self.max_final_margin_range),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if type(self.require_final_winner_unanimity) is not bool:
            raise ValueError("require_final_winner_unanimity must be boolean")

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)


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
            right_value - left_value for left_value, right_value in zip(left_trace, right_trace, strict=True)
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
            "mean_abs_margin_difference": mean(float(row["mean_abs_margin_difference"]) for row in pairs),
            "all_final_pairwise_winners_agree": all(
                bool(row["final_pairwise_winner_agreement"]) for row in pairs
            ),
        },
    }


def compare_artifacts(left: dict, right: dict) -> dict[str, object]:
    left_experiment = left.get("experiment")
    right_experiment = right.get("experiment")
    if not isinstance(left_experiment, dict) or not isinstance(right_experiment, dict):
        raise TypeError("both artifacts must contain experiment metadata dictionaries")

    identity_fields = ("prompt", "candidates", "model_name")
    mismatches = [
        field for field in identity_fields if left_experiment.get(field) != right_experiment.get(field)
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


def _validated_cohort_artifact(
    artifact: object,
    labels: tuple[str, ...],
    expected_identity: dict[str, object],
    expected_fingerprint: str,
) -> dict[str, object]:
    if not isinstance(artifact, dict):
        raise StabilityGateError("each run artifact must be a dictionary")
    experiment = artifact.get("experiment")
    provenance = artifact.get("provenance")
    analysis = artifact.get("base")
    if not isinstance(experiment, dict) or not isinstance(provenance, dict):
        raise StabilityGateError("each run must contain experiment and provenance dictionaries")
    if any(experiment.get(name) != value for name, value in expected_identity.items()):
        raise StabilityGateError("run artifacts describe different experiments")
    fingerprint = provenance.get("experiment_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or _DIGEST_PATTERN.fullmatch(fingerprint) is None
        or fingerprint != expected_fingerprint
    ):
        raise StabilityGateError("run artifacts have missing or different fingerprints")
    if not isinstance(analysis, dict):
        raise StabilityGateError("each run must contain a base analysis dictionary")
    layers = analysis.get("layers")
    if not isinstance(layers, list) or not 1 <= len(layers) <= _MAX_LAYERS:
        raise StabilityGateError("each base analysis must contain a bounded non-empty layer list")
    previous_layer: int | None = None
    for row in layers:
        if not isinstance(row, dict) or type(row.get("layer")) is not int:
            raise StabilityGateError("layer rows must contain integer layer indices")
        layer = row["layer"]
        if layer < 0 or (previous_layer is not None and layer <= previous_layer):
            raise StabilityGateError("layer indices must be non-negative and strictly increasing")
        scores = row.get("candidate_scores")
        if not isinstance(scores, dict) or set(scores) != set(labels):
            raise StabilityGateError("every layer must contain the exact candidate score set")
        for score in scores.values():
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                raise StabilityGateError("candidate scores must be finite numbers")
        previous_layer = layer
    return analysis


def audit_artifact_stability_cohort(
    artifacts: Mapping[str, dict],
    *,
    policy: StabilityCohortPolicy | None = None,
) -> dict[str, object]:
    """Gate one experiment's base decision trace across a bounded cohort of runs."""

    active_policy = policy or StabilityCohortPolicy()
    if not isinstance(active_policy, StabilityCohortPolicy):
        raise TypeError("policy must be a StabilityCohortPolicy")
    if not isinstance(artifacts, Mapping):
        raise TypeError("artifacts must be a mapping")
    if not active_policy.min_runs <= len(artifacts) <= active_policy.max_runs:
        raise StabilityGateError("run count is outside the policy bounds")
    run_ids = tuple(artifacts)
    if any(not isinstance(run_id, str) or _RUN_ID_PATTERN.fullmatch(run_id) is None for run_id in run_ids):
        raise StabilityGateError("run IDs must be canonical identifiers")
    ordered_ids = tuple(sorted(run_ids))
    first_artifact = artifacts[ordered_ids[0]]
    if not isinstance(first_artifact, dict):
        raise StabilityGateError("each run artifact must be a dictionary")
    first_experiment = first_artifact.get("experiment")
    first_provenance = first_artifact.get("provenance")
    if not isinstance(first_experiment, dict) or not isinstance(first_provenance, dict):
        raise StabilityGateError("each run must contain experiment and provenance dictionaries")
    identity = {name: first_experiment.get(name) for name in ("prompt", "candidates", "model_name")}
    if not isinstance(identity["prompt"], str) or not identity["prompt"]:
        raise StabilityGateError("experiment prompt must be a non-empty string")
    if not isinstance(identity["model_name"], str) or not identity["model_name"]:
        raise StabilityGateError("experiment model_name must be a non-empty string")
    candidates = identity["candidates"]
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= 16:
        raise StabilityGateError("experiment must contain from 2 through 16 candidates")
    labels = tuple(candidates)
    if any(
        not isinstance(label, str)
        or not label
        or len(label.encode("utf-8")) > 256
        or any(ord(character) < 32 for character in label)
        for label in labels
    ) or len(set(labels)) != len(labels):
        raise StabilityGateError("candidate labels must be unique bounded strings")
    fingerprint = first_provenance.get("experiment_fingerprint")
    if not isinstance(fingerprint, str) or _DIGEST_PATTERN.fullmatch(fingerprint) is None:
        raise StabilityGateError("experiment fingerprint must be a canonical SHA-256")

    analyses: dict[str, dict[str, object]] = {}
    run_commitments: list[dict[str, str]] = []
    layer_indices: tuple[int, ...] | None = None
    for run_id in ordered_ids:
        artifact = artifacts[run_id]
        analysis = _validated_cohort_artifact(artifact, labels, identity, fingerprint)
        current_layers = tuple(_layer_indices(analysis))
        if layer_indices is None:
            layer_indices = current_layers
        elif current_layers != layer_indices:
            raise StabilityGateError("run artifacts contain different layer indices")
        analyses[run_id] = analysis
        run_commitments.append(
            {
                "run_id": run_id,
                "run_sha256": _canonical_digest(
                    {
                        "experiment": identity,
                        "experiment_fingerprint": fingerprint,
                        "base": analysis,
                    }
                ),
            }
        )
    assert layer_indices is not None
    comparison_count = len(ordered_ids) * (len(ordered_ids) - 1) // 2
    candidate_pair_count = len(labels) * (len(labels) - 1) // 2
    work_units = comparison_count * candidate_pair_count * len(layer_indices)
    if work_units > 5_000_000:
        raise StabilityGateError("cohort comparison exceeds the work budget")

    pairwise_results = [
        compare_runs(analyses[left_id], analyses[right_id], labels)
        for left_id, right_id in combinations(ordered_ids, 2)
    ]
    pair_rows: list[dict[str, object]] = []
    global_findings: set[str] = set()
    for pair_index, (positive, negative) in enumerate(combinations(labels, 2)):
        comparisons_for_pair = [result["pairs"][pair_index] for result in pairwise_results]
        correlations = [float(row["pearson_correlation"]) for row in comparisons_for_pair]
        sign_agreements = [float(row["sign_agreement_rate"]) for row in comparisons_for_pair]
        mean_differences = [float(row["mean_abs_margin_difference"]) for row in comparisons_for_pair]
        final_margins = [_margin_trace(analyses[run_id], positive, negative)[-1] for run_id in ordered_ids]
        final_signs = {_sign(value) for value in final_margins}
        findings: list[str] = []
        if min(correlations) < active_policy.min_pairwise_pearson:
            findings.append("PAIRWISE_PEARSON_BELOW_FLOOR")
        if min(sign_agreements) < active_policy.min_pairwise_sign_agreement:
            findings.append("PAIRWISE_SIGN_AGREEMENT_BELOW_FLOOR")
        if max(mean_differences) > active_policy.max_pairwise_mean_abs_margin_difference:
            findings.append("PAIRWISE_MARGIN_DRIFT_ABOVE_LIMIT")
        final_margin_range = max(final_margins) - min(final_margins)
        if final_margin_range > active_policy.max_final_margin_range:
            findings.append("FINAL_MARGIN_RANGE_ABOVE_LIMIT")
        final_winner_unanimous = len(final_signs) == 1
        if active_policy.require_final_winner_unanimity and not final_winner_unanimous:
            findings.append("FINAL_WINNER_DISAGREEMENT")
        global_findings.update(findings)
        pair_rows.append(
            {
                "positive": positive,
                "negative": negative,
                "mean_pairwise_pearson": mean(correlations),
                "minimum_pairwise_pearson": min(correlations),
                "mean_pairwise_sign_agreement": mean(sign_agreements),
                "minimum_pairwise_sign_agreement": min(sign_agreements),
                "maximum_pairwise_mean_abs_margin_difference": max(mean_differences),
                "final_margin_range": final_margin_range,
                "final_winner_unanimous": final_winner_unanimous,
                "qualifies": not findings,
                "findings": findings,
            }
        )

    policy_document = active_policy.to_audit_dict()
    report: dict[str, object] = {
        "schema_version": STABILITY_COHORT_SCHEMA,
        "experiment_sha256": _canonical_digest(identity),
        "experiment_fingerprint": fingerprint,
        "policy": policy_document,
        "policy_sha256": _canonical_digest(policy_document),
        "run_count": len(ordered_ids),
        "pairwise_comparison_count": comparison_count,
        "work_units": work_units,
        "runs": run_commitments,
        "pairs": pair_rows,
        "accepted": not global_findings,
        "finding_codes": sorted(global_findings),
    }
    report["evidence_sha256"] = _canonical_digest(report)
    return report


def require_artifact_stability_cohort(
    artifacts: Mapping[str, dict],
    *,
    policy: StabilityCohortPolicy | None = None,
) -> dict[str, object]:
    """Return accepted evidence or raise one stable scientific gate error."""

    report = audit_artifact_stability_cohort(artifacts, policy=policy)
    if not report["accepted"]:
        findings = report["finding_codes"]
        assert isinstance(findings, list)
        raise StabilityGateError("policy failed: " + ",".join(str(item) for item in findings))
    return report
