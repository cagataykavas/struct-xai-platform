"""Audit candidate-swap equivariance for pairwise attribution artifacts.

The audit is model-free by design: production or research runners can emit the
small JSON contract documented in ``docs/candidate-swap-audit.md`` and check it
without loading the model again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class CandidateSwapInputError(ValueError):
    """Raised when the evidence or policy cannot be audited safely."""


@dataclass(frozen=True)
class CandidateSwapPolicy:
    """Release thresholds and resource limits for a swap audit."""

    min_cases: int = 3
    min_features_per_case: int = 3
    top_k: int = 3
    min_margin_magnitude: float = 1e-6
    max_margin_antisymmetry_error: float = 0.05
    max_attribution_antisymmetry_error: float = 0.15
    min_anti_cosine_similarity: float = 0.95
    min_top_k_overlap: float = 0.80
    min_case_pass_rate: float = 1.0
    zero_tolerance: float = 1e-12
    max_cases: int = 10_000
    max_features_per_case: int = 10_000
    max_total_features: int = 1_000_000
    max_identifier_length: int = 256


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateSwapInputError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CandidateSwapInputError(f"{field} must be a finite number")
    return result


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CandidateSwapInputError(f"{field} must be a positive integer")
    return value


def _validate_policy(policy: CandidateSwapPolicy) -> None:
    for field in (
        "min_cases",
        "min_features_per_case",
        "top_k",
        "max_cases",
        "max_features_per_case",
        "max_total_features",
        "max_identifier_length",
    ):
        _positive_int(getattr(policy, field), f"policy.{field}")

    if policy.min_cases > policy.max_cases:
        raise CandidateSwapInputError("policy.min_cases cannot exceed policy.max_cases")
    if policy.top_k > policy.min_features_per_case:
        raise CandidateSwapInputError("policy.top_k cannot exceed policy.min_features_per_case")
    if policy.min_features_per_case > policy.max_features_per_case:
        raise CandidateSwapInputError(
            "policy.min_features_per_case cannot exceed policy.max_features_per_case"
        )

    for field in (
        "max_margin_antisymmetry_error",
        "max_attribution_antisymmetry_error",
        "min_anti_cosine_similarity",
        "min_top_k_overlap",
        "min_case_pass_rate",
    ):
        value = _finite_number(getattr(policy, field), f"policy.{field}")
        if not 0.0 <= value <= 1.0:
            raise CandidateSwapInputError(f"policy.{field} must be between 0 and 1")

    for field in ("min_margin_magnitude", "zero_tolerance"):
        value = _finite_number(getattr(policy, field), f"policy.{field}")
        if value <= 0.0:
            raise CandidateSwapInputError(f"policy.{field} must be greater than zero")


def _identifier(value: Any, field: str, policy: CandidateSwapPolicy) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateSwapInputError(f"{field} must be a non-empty string")
    if len(value) > policy.max_identifier_length:
        raise CandidateSwapInputError(
            f"{field} exceeds policy.max_identifier_length={policy.max_identifier_length}"
        )
    return value


def _attribution_map(value: Any, field: str, policy: CandidateSwapPolicy) -> dict[str, float]:
    if not isinstance(value, list):
        raise CandidateSwapInputError(f"{field} must be a list")
    if len(value) < policy.min_features_per_case:
        raise CandidateSwapInputError(
            f"{field} must contain at least {policy.min_features_per_case} features"
        )
    if len(value) > policy.max_features_per_case:
        raise CandidateSwapInputError(
            f"{field} exceeds policy.max_features_per_case={policy.max_features_per_case}"
        )

    result: dict[str, float] = {}
    for index, row in enumerate(value):
        item_field = f"{field}[{index}]"
        if not isinstance(row, dict):
            raise CandidateSwapInputError(f"{item_field} must be an object")
        feature_id = _identifier(row.get("feature_id"), f"{item_field}.feature_id", policy)
        if feature_id in result:
            raise CandidateSwapInputError(f"{field} contains duplicate feature_id {feature_id!r}")
        result[feature_id] = _finite_number(row.get("value"), f"{item_field}.value")
    return result


def _observation(
    value: Any, field: str, expected_order: tuple[str, str], policy: CandidateSwapPolicy
) -> tuple[float, dict[str, float]]:
    if not isinstance(value, dict):
        raise CandidateSwapInputError(f"{field} must be an object")
    order = value.get("candidate_order")
    if not isinstance(order, list) or tuple(order) != expected_order:
        raise CandidateSwapInputError(f"{field}.candidate_order must be {list(expected_order)!r}")
    margin = _finite_number(value.get("margin"), f"{field}.margin")
    attributions = _attribution_map(value.get("attributions"), f"{field}.attributions", policy)
    return margin, attributions


def _top_k(values: dict[str, float], k: int) -> set[str]:
    ranked = sorted(values, key=lambda key: (-abs(values[key]), key))
    return set(ranked[:k])


def _case_metrics(
    forward_margin: float,
    swapped_margin: float,
    forward: dict[str, float],
    swapped: dict[str, float],
    policy: CandidateSwapPolicy,
) -> dict[str, float]:
    ordered_ids = sorted(forward)
    forward_values = [forward[feature_id] for feature_id in ordered_ids]
    swapped_values = [swapped[feature_id] for feature_id in ordered_ids]

    forward_l1 = sum(abs(value) for value in forward_values)
    swapped_l1 = sum(abs(value) for value in swapped_values)
    if forward_l1 <= policy.zero_tolerance or swapped_l1 <= policy.zero_tolerance:
        raise CandidateSwapInputError("attribution vectors must have non-zero L1 norm")

    forward_l2 = math.sqrt(sum(value * value for value in forward_values))
    swapped_l2 = math.sqrt(sum(value * value for value in swapped_values))
    anti_dot = sum(left * -right for left, right in zip(forward_values, swapped_values, strict=True))
    anti_cosine = anti_dot / (forward_l2 * swapped_l2)
    attribution_error = sum(
        abs(left + right) for left, right in zip(forward_values, swapped_values, strict=True)
    ) / max(forward_l1, swapped_l1)
    margin_error = abs(forward_margin + swapped_margin) / max(
        abs(forward_margin), abs(swapped_margin), policy.zero_tolerance
    )
    top_k_overlap = len(_top_k(forward, policy.top_k) & _top_k(swapped, policy.top_k)) / policy.top_k

    return {
        "forward_margin": forward_margin,
        "swapped_margin": swapped_margin,
        "margin_antisymmetry_error": margin_error,
        "attribution_antisymmetry_error": attribution_error,
        "anti_cosine_similarity": anti_cosine,
        "top_k_overlap": top_k_overlap,
        "feature_count": len(forward),
    }


def _normalized_artifact(cases: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_cases = []
    for case in sorted(cases, key=lambda item: item["case_id"]):
        normalized_cases.append(
            {
                "case_id": case["case_id"],
                "candidate_a": case["candidate_a"],
                "candidate_b": case["candidate_b"],
                "forward_margin": case["forward_margin"],
                "swapped_margin": case["swapped_margin"],
                "forward": sorted(case["forward"].items()),
                "swapped": sorted(case["swapped"].items()),
            }
        )
    return {"schema_version": 1, "cases": normalized_cases}


def audit_candidate_swaps(
    artifact: dict[str, Any], policy: CandidateSwapPolicy | None = None
) -> dict[str, Any]:
    """Validate and audit pairwise candidate-swap attribution evidence.

    Malformed or incomplete evidence raises :class:`CandidateSwapInputError`.
    Well-formed evidence returns a JSON-ready release decision.
    """

    active_policy = policy or CandidateSwapPolicy()
    _validate_policy(active_policy)
    if not isinstance(artifact, dict):
        raise CandidateSwapInputError("artifact must be an object")
    if artifact.get("schema_version") != 1:
        raise CandidateSwapInputError("artifact.schema_version must equal 1")
    rows = artifact.get("cases")
    if not isinstance(rows, list):
        raise CandidateSwapInputError("artifact.cases must be a list")
    if len(rows) < active_policy.min_cases:
        raise CandidateSwapInputError(f"artifact.cases must contain at least {active_policy.min_cases} cases")
    if len(rows) > active_policy.max_cases:
        raise CandidateSwapInputError(f"artifact.cases exceeds policy.max_cases={active_policy.max_cases}")

    parsed: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    total_features = 0
    for index, row in enumerate(rows):
        field = f"artifact.cases[{index}]"
        if not isinstance(row, dict):
            raise CandidateSwapInputError(f"{field} must be an object")
        case_id = _identifier(row.get("case_id"), f"{field}.case_id", active_policy)
        if case_id in case_ids:
            raise CandidateSwapInputError(f"artifact contains duplicate case_id {case_id!r}")
        case_ids.add(case_id)
        candidate_a = _identifier(row.get("candidate_a"), f"{field}.candidate_a", active_policy)
        candidate_b = _identifier(row.get("candidate_b"), f"{field}.candidate_b", active_policy)
        if candidate_a == candidate_b:
            raise CandidateSwapInputError(f"{field} candidates must be distinct")

        forward_margin, forward = _observation(
            row.get("forward"),
            f"{field}.forward",
            (candidate_a, candidate_b),
            active_policy,
        )
        swapped_margin, swapped = _observation(
            row.get("swapped"),
            f"{field}.swapped",
            (candidate_b, candidate_a),
            active_policy,
        )
        if set(forward) != set(swapped):
            missing = sorted(set(forward) - set(swapped))
            unexpected = sorted(set(swapped) - set(forward))
            raise CandidateSwapInputError(
                f"{field} feature IDs do not align; missing={missing!r}, unexpected={unexpected!r}"
            )
        total_features += len(forward)
        if total_features > active_policy.max_total_features:
            raise CandidateSwapInputError(
                f"artifact exceeds policy.max_total_features={active_policy.max_total_features}"
            )
        parsed.append(
            {
                "case_id": case_id,
                "candidate_a": candidate_a,
                "candidate_b": candidate_b,
                "forward_margin": forward_margin,
                "swapped_margin": swapped_margin,
                "forward": forward,
                "swapped": swapped,
            }
        )

    results: list[dict[str, Any]] = []
    finding_counts: Counter[str] = Counter()
    for case in sorted(parsed, key=lambda item: item["case_id"]):
        metrics = _case_metrics(
            case["forward_margin"],
            case["swapped_margin"],
            case["forward"],
            case["swapped"],
            active_policy,
        )
        findings: list[str] = []
        if min(abs(metrics["forward_margin"]), abs(metrics["swapped_margin"])) < (
            active_policy.min_margin_magnitude
        ):
            findings.append("WEAK_PAIRWISE_MARGIN")
        if metrics["margin_antisymmetry_error"] > active_policy.max_margin_antisymmetry_error:
            findings.append("MARGIN_ANTISYMMETRY_FAILED")
        if metrics["attribution_antisymmetry_error"] > active_policy.max_attribution_antisymmetry_error:
            findings.append("ATTRIBUTION_ANTISYMMETRY_FAILED")
        if metrics["anti_cosine_similarity"] < active_policy.min_anti_cosine_similarity:
            findings.append("ANTI_COSINE_FAILED")
        if metrics["top_k_overlap"] < active_policy.min_top_k_overlap:
            findings.append("TOP_K_OVERLAP_FAILED")
        finding_counts.update(findings)
        results.append(
            {
                "case_id": case["case_id"],
                "candidate_a": case["candidate_a"],
                "candidate_b": case["candidate_b"],
                "passed": not findings,
                "findings": findings,
                "metrics": metrics,
            }
        )

    passed_cases = sum(result["passed"] for result in results)
    case_pass_rate = passed_cases / len(results)
    release_findings = []
    if case_pass_rate < active_policy.min_case_pass_rate:
        release_findings.append("CASE_PASS_RATE_BELOW_POLICY")
    accepted = not release_findings
    canonical = json.dumps(
        _normalized_artifact(parsed), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")

    return {
        "audit": "candidate_swap_attribution_equivariance",
        "schema_version": 1,
        "accepted": accepted,
        "release_findings": release_findings,
        "policy": asdict(active_policy),
        "evidence": {
            "artifact_sha256": hashlib.sha256(canonical).hexdigest(),
            "case_count": len(results),
            "total_feature_count": total_features,
        },
        "summary": {
            "passed_cases": passed_cases,
            "failed_cases": len(results) - passed_cases,
            "case_pass_rate": case_pass_rate,
            "finding_counts": dict(sorted(finding_counts.items())),
        },
        "cases": results,
    }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CandidateSwapInputError(f"JSON contains duplicate field {key!r}")
        result[key] = value
    return result


def load_artifact(path: Path, max_bytes: int = 16 * 1024 * 1024) -> dict[str, Any]:
    """Load bounded JSON while rejecting duplicate object fields."""

    if path.stat().st_size > max_bytes:
        raise CandidateSwapInputError(f"artifact exceeds {max_bytes} bytes")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateSwapInputError(f"cannot read artifact: {exc}") from exc
    if not isinstance(value, dict):
        raise CandidateSwapInputError("artifact must be a JSON object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit candidate-swap equivariance of pairwise attribution evidence."
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-cases", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-features", type=int, default=3)
    parser.add_argument("--max-margin-error", type=float, default=0.05)
    parser.add_argument("--max-attribution-error", type=float, default=0.15)
    parser.add_argument("--min-anti-cosine", type=float, default=0.95)
    parser.add_argument("--min-top-k-overlap", type=float, default=0.80)
    parser.add_argument("--min-case-pass-rate", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        policy = CandidateSwapPolicy(
            min_cases=args.min_cases,
            min_features_per_case=args.min_features,
            top_k=args.top_k,
            max_margin_antisymmetry_error=args.max_margin_error,
            max_attribution_antisymmetry_error=args.max_attribution_error,
            min_anti_cosine_similarity=args.min_anti_cosine,
            min_top_k_overlap=args.min_top_k_overlap,
            min_case_pass_rate=args.min_case_pass_rate,
        )
        report = audit_candidate_swaps(load_artifact(args.artifact), policy)
    except (CandidateSwapInputError, OSError) as exc:
        print(
            json.dumps(
                {"status": "invalid_input", "error_code": "INVALID_SWAP_EVIDENCE", "message": str(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 3

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
