"""Fail-closed audit for attribution stability across random seeds.

The audit consumes model-independent evidence.  It deliberately reports only
hashed case identities and aggregate metrics so CI artifacts need not disclose
prompts, tokens, or attribution values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any, NoReturn

SCHEMA = "struct-xai-attribution-seed-stability/v1"
MAX_ARTIFACT_BYTES = 1_048_576
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class ArtifactError(ValueError):
    """Raised when evidence is malformed or exceeds a resource budget."""


@dataclass(frozen=True)
class StabilityPolicy:
    min_cases: int = 1
    min_runs_per_case: int = 5
    top_k: int = 3
    min_pairwise_cosine: float = 0.90
    min_pairwise_top_k_jaccard: float = 0.80
    min_material_sign_agreement: float = 0.90
    min_consensus_top_k_rate: float = 0.80
    max_output_margin_span: float = 1e-9
    max_failed_case_fraction: float = 0.0
    max_cases: int = 256
    max_runs_per_case: int = 32
    max_features_per_case: int = 4096
    max_pair_comparisons: int = 50_000
    max_artifact_bytes: int = MAX_ARTIFACT_BYTES
    max_age_seconds: int = 7 * 24 * 60 * 60
    max_future_skew_seconds: int = 300
    zero_epsilon: float = 1e-12

    def validate(self) -> None:
        _bounded_int("min_cases", self.min_cases, 1, self.max_cases)
        _bounded_int("min_runs_per_case", self.min_runs_per_case, 2, self.max_runs_per_case)
        _bounded_int("top_k", self.top_k, 1, self.max_features_per_case)
        _bounded_int("max_cases", self.max_cases, 1, 10_000)
        _bounded_int("max_runs_per_case", self.max_runs_per_case, 2, 1_000)
        _bounded_int("max_features_per_case", self.max_features_per_case, 1, 100_000)
        _bounded_int("max_pair_comparisons", self.max_pair_comparisons, 1, 10_000_000)
        _bounded_int("max_artifact_bytes", self.max_artifact_bytes, 1, 32 * 1024 * 1024)
        _bounded_int("max_age_seconds", self.max_age_seconds, 1, 365 * 24 * 60 * 60)
        _bounded_int("max_future_skew_seconds", self.max_future_skew_seconds, 0, 86_400)
        for name in (
            "min_pairwise_cosine",
            "min_pairwise_top_k_jaccard",
            "min_material_sign_agreement",
            "min_consensus_top_k_rate",
        ):
            _bounded_float(name, getattr(self, name), 0.0, 1.0)
        _bounded_float("max_failed_case_fraction", self.max_failed_case_fraction, 0.0, 1.0)
        _bounded_float("max_output_margin_span", self.max_output_margin_span, 0.0, 1e12)
        _bounded_float("zero_epsilon", self.zero_epsilon, 0.0, 1.0, lower_inclusive=False)


def _fail(message: str) -> NoReturn:
    raise ArtifactError(message)


def _bounded_int(name: str, value: object, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        _fail(f"{name} must be an integer in [{lower}, {upper}]")
    return value


def _bounded_float(
    name: str,
    value: object,
    lower: float,
    upper: float,
    *,
    lower_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{name} must be numeric")
    result = float(value)
    lower_ok = result >= lower if lower_inclusive else result > lower
    if not math.isfinite(result) or not lower_ok or result > upper:
        bracket = "[" if lower_inclusive else "("
        _fail(f"{name} must be finite and in {bracket}{lower}, {upper}]")
    return result


def _object(value: object, name: str, required: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != required:
        _fail(f"{name} must contain exactly: {', '.join(sorted(required))}")
    if not all(isinstance(key, str) for key in value):
        _fail(f"{name} keys must be strings")
    return value


def _identifier(value: object, name: str, *, max_length: int = 128) -> str:
    if not isinstance(value, str) or not value or len(value) > max_length:
        _fail(f"{name} must be a non-empty string of at most {max_length} characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        _fail(f"{name} must not contain control characters")
    return value


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        _fail(f"{name} must be a lowercase SHA-256 digest")
    return value


def _timestamp(value: object, name: str) -> datetime:
    if not isinstance(value, str) or len(value) > 64:
        _fail(f"{name} must be a bounded ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ArtifactError(f"{name} must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail(f"{name} must include a timezone")
    return parsed.astimezone(UTC)


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArtifactError("artifact must be canonical JSON with finite values") from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _hash_identifier(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sign(value: float, epsilon: float) -> int:
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def _cosine(left: list[float], right: list[float], epsilon: float) -> float:
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm <= epsilon or right_norm <= epsilon:
        _fail("every attribution run must have a non-zero vector")
    value = math.fsum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    return max(-1.0, min(1.0, value))


def _top_ids(feature_ids: list[str], values: list[float], top_k: int) -> set[str]:
    ranked = sorted(zip(feature_ids, values, strict=True), key=lambda row: (-abs(row[1]), row[0]))
    return {feature_id for feature_id, _ in ranked[:top_k]}


def _parse_run(
    payload: object,
    *,
    case_index: int,
    run_index: int,
    policy: StabilityPolicy,
) -> tuple[int, float, dict[str, float]]:
    prefix = f"cases[{case_index}].runs[{run_index}]"
    run = _object(payload, prefix, {"seed", "output_margin", "features"})
    seed = _bounded_int(f"{prefix}.seed", run["seed"], -(2**63), 2**63 - 1)
    margin = _bounded_float(f"{prefix}.output_margin", run["output_margin"], -1e12, 1e12)
    features = run["features"]
    if not isinstance(features, list) or not features:
        _fail(f"{prefix}.features must be a non-empty list")
    if len(features) > policy.max_features_per_case:
        _fail(f"{prefix}.features exceeds max_features_per_case")

    parsed: dict[str, float] = {}
    for feature_index, raw_feature in enumerate(features):
        feature_name = f"{prefix}.features[{feature_index}]"
        feature = _object(raw_feature, feature_name, {"feature_id", "attribution"})
        feature_id = _identifier(feature["feature_id"], f"{feature_name}.feature_id")
        if feature_id in parsed:
            _fail(f"duplicate feature_id in {prefix}")
        parsed[feature_id] = _bounded_float(
            f"{feature_name}.attribution", feature["attribution"], -1e12, 1e12
        )
    return seed, margin, parsed


def _case_metrics(
    payload: object,
    *,
    case_index: int,
    policy: StabilityPolicy,
) -> tuple[dict[str, object], int]:
    case = _object(payload, f"cases[{case_index}]", {"case_id", "runs"})
    case_id = _identifier(case["case_id"], f"cases[{case_index}].case_id")
    runs = case["runs"]
    if not isinstance(runs, list):
        _fail(f"cases[{case_index}].runs must be a list")
    if not policy.min_runs_per_case <= len(runs) <= policy.max_runs_per_case:
        _fail(f"cases[{case_index}].runs is outside the configured run bounds")

    parsed = [
        _parse_run(run, case_index=case_index, run_index=index, policy=policy)
        for index, run in enumerate(runs)
    ]
    seeds = [seed for seed, _, _ in parsed]
    if len(seeds) != len(set(seeds)):
        _fail(f"cases[{case_index}] contains duplicate seeds")

    feature_sets = [set(features) for _, _, features in parsed]
    if any(feature_set != feature_sets[0] for feature_set in feature_sets[1:]):
        _fail(f"cases[{case_index}] runs must contain the exact same feature IDs")
    feature_ids = sorted(feature_sets[0])
    if policy.top_k > len(feature_ids):
        _fail(f"cases[{case_index}] has fewer features than top_k")
    vectors = [[features[feature_id] for feature_id in feature_ids] for _, _, features in parsed]
    top_sets = [_top_ids(feature_ids, vector, policy.top_k) for vector in vectors]

    cosines: list[float] = []
    overlaps: list[float] = []
    sign_agreements: list[float] = []
    for left_index, right_index in combinations(range(len(vectors)), 2):
        left = vectors[left_index]
        right = vectors[right_index]
        cosines.append(_cosine(left, right, policy.zero_epsilon))
        left_top = top_sets[left_index]
        right_top = top_sets[right_index]
        overlaps.append(len(left_top & right_top) / len(left_top | right_top))
        material = sorted(left_top | right_top)
        matches = sum(
            _sign(left[feature_ids.index(feature_id)], policy.zero_epsilon)
            == _sign(right[feature_ids.index(feature_id)], policy.zero_epsilon)
            for feature_id in material
        )
        sign_agreements.append(matches / len(material))

    selections = {feature_id: 0 for feature_id in feature_ids}
    for top_set in top_sets:
        for feature_id in top_set:
            selections[feature_id] += 1
    consensus_rate = math.fsum(sorted(selections.values(), reverse=True)[: policy.top_k]) / (
        len(runs) * policy.top_k
    )
    margins = [margin for _, margin, _ in parsed]
    margin_span = max(margins) - min(margins)

    metrics = {
        "min_pairwise_cosine": min(cosines),
        "mean_pairwise_cosine": math.fsum(cosines) / len(cosines),
        "min_pairwise_top_k_jaccard": min(overlaps),
        "mean_pairwise_top_k_jaccard": math.fsum(overlaps) / len(overlaps),
        "min_material_sign_agreement": min(sign_agreements),
        "consensus_top_k_rate": consensus_rate,
        "output_margin_span": margin_span,
    }
    reasons: list[str] = []
    if metrics["min_pairwise_cosine"] < policy.min_pairwise_cosine:
        reasons.append("pairwise_cosine_below_threshold")
    if metrics["min_pairwise_top_k_jaccard"] < policy.min_pairwise_top_k_jaccard:
        reasons.append("top_k_overlap_below_threshold")
    if metrics["min_material_sign_agreement"] < policy.min_material_sign_agreement:
        reasons.append("material_sign_agreement_below_threshold")
    if metrics["consensus_top_k_rate"] < policy.min_consensus_top_k_rate:
        reasons.append("consensus_top_k_rate_below_threshold")
    if metrics["output_margin_span"] > policy.max_output_margin_span:
        reasons.append("output_margin_drift_above_threshold")

    return (
        {
            "case_sha256": _hash_identifier(case_id),
            "passed": not reasons,
            "reasons": reasons,
            "run_count": len(runs),
            "feature_count": len(feature_ids),
            "pair_comparisons": len(cosines),
            "metrics": metrics,
        },
        len(cosines),
    )


def audit_artifact(
    payload: object,
    *,
    policy: StabilityPolicy | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """Validate and audit one attribution seed-stability artifact."""
    active_policy = policy or StabilityPolicy()
    active_policy.validate()
    artifact_bytes = _canonical_bytes(payload)
    if len(artifact_bytes) > active_policy.max_artifact_bytes:
        _fail("artifact exceeds max_artifact_bytes")

    root = _object(
        payload,
        "artifact",
        {
            "schema",
            "generated_at",
            "benchmark_id",
            "model_sha256",
            "method_id",
            "method_config_sha256",
            "cases",
        },
    )
    if root["schema"] != SCHEMA:
        _fail(f"schema must equal {SCHEMA}")
    generated_at = _timestamp(root["generated_at"], "generated_at")
    current = (now or datetime.now(UTC)).astimezone(UTC)
    age = (current - generated_at).total_seconds()
    if age > active_policy.max_age_seconds:
        _fail("artifact is stale")
    if age < -active_policy.max_future_skew_seconds:
        _fail("artifact timestamp is too far in the future")

    benchmark_id = _identifier(root["benchmark_id"], "benchmark_id")
    model_digest = _digest(root["model_sha256"], "model_sha256")
    method_id = _identifier(root["method_id"], "method_id")
    method_config_digest = _digest(root["method_config_sha256"], "method_config_sha256")
    cases = root["cases"]
    if not isinstance(cases, list):
        _fail("cases must be a list")
    if not active_policy.min_cases <= len(cases) <= active_policy.max_cases:
        _fail("cases is outside the configured case bounds")

    results: list[dict[str, object]] = []
    seen_case_hashes: set[str] = set()
    comparison_count = 0
    for index, case in enumerate(cases):
        result, comparisons = _case_metrics(case, case_index=index, policy=active_policy)
        case_hash = str(result["case_sha256"])
        if case_hash in seen_case_hashes:
            _fail("case_id values must be unique")
        seen_case_hashes.add(case_hash)
        comparison_count += comparisons
        if comparison_count > active_policy.max_pair_comparisons:
            _fail("artifact exceeds max_pair_comparisons")
        results.append(result)

    failed = sum(not bool(result["passed"]) for result in results)
    failed_fraction = failed / len(results)
    accepted = failed_fraction <= active_policy.max_failed_case_fraction
    policy_payload = asdict(active_policy)
    evidence_payload = {
        "schema": SCHEMA,
        "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "policy_sha256": _sha256(policy_payload),
        "cases": results,
    }
    return {
        "schema": "struct-xai-attribution-seed-stability-report/v1",
        "accepted": accepted,
        "reason": "accepted" if accepted else "seed_stability_policy_rejected",
        "identity": {
            "benchmark_sha256": _hash_identifier(benchmark_id),
            "model_sha256": model_digest,
            "method_sha256": _hash_identifier(method_id),
            "method_config_sha256": method_config_digest,
        },
        "summary": {
            "case_count": len(results),
            "failed_case_count": failed,
            "failed_case_fraction": failed_fraction,
            "pair_comparisons": comparison_count,
        },
        "policy": policy_payload,
        "cases": results,
        "artifact_sha256": evidence_payload["artifact_sha256"],
        "policy_sha256": evidence_payload["policy_sha256"],
        "evidence_sha256": _sha256(evidence_payload),
    }


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_artifact(path: Path, *, max_bytes: int = MAX_ARTIFACT_BYTES) -> object:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ArtifactError("unable to read artifact") from exc
    if len(raw) > max_bytes:
        _fail("artifact exceeds the input byte budget")
    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: _fail(f"non-finite JSON value: {value}"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError("artifact must be valid UTF-8 JSON") from exc


def write_report_atomic(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-runs", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-cosine", type=float, default=0.90)
    parser.add_argument("--min-top-k-jaccard", type=float, default=0.80)
    parser.add_argument("--min-sign-agreement", type=float, default=0.90)
    parser.add_argument("--min-consensus-rate", type=float, default=0.80)
    parser.add_argument("--max-margin-span", type=float, default=1e-9)
    parser.add_argument("--max-failed-case-fraction", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        policy = StabilityPolicy(
            min_runs_per_case=args.min_runs,
            top_k=args.top_k,
            min_pairwise_cosine=args.min_cosine,
            min_pairwise_top_k_jaccard=args.min_top_k_jaccard,
            min_material_sign_agreement=args.min_sign_agreement,
            min_consensus_top_k_rate=args.min_consensus_rate,
            max_output_margin_span=args.max_margin_span,
            max_failed_case_fraction=args.max_failed_case_fraction,
        )
        artifact = load_artifact(args.artifact, max_bytes=policy.max_artifact_bytes)
        report = audit_artifact(artifact, policy=policy)
    except ArtifactError as exc:
        print(json.dumps({"accepted": False, "reason": "malformed_artifact", "error": str(exc)}))
        return 3

    if args.output is not None:
        write_report_atomic(args.output, report)
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
