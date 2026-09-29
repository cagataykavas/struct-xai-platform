"""Audit token-attribution consistency across tokenizer migrations.

Token scores from different segmentations are projected onto a shared UTF-8
byte axis. The audit is deliberately model-independent and consumes bounded
evidence exported by an explanation runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARTIFACT_SCHEMA = "struct-xai/tokenizer-migration/v1"
REPORT_SCHEMA = "struct-xai/tokenizer-migration-report/v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


class MalformedArtifact(ValueError):
    """Raised when migration evidence cannot be evaluated safely."""


@dataclass(frozen=True)
class TokenizerMigrationPolicy:
    min_signed_cosine: float = 0.90
    min_top_k_jaccard: float = 0.70
    max_relative_l1: float = 0.30
    max_output_margin_drift: float = 1e-4
    max_attribution_sum_drift: float = 1e-5
    max_failed_case_fraction: float = 0.0
    top_k_fraction: float = 0.10
    min_vector_l1: float = 1e-8
    max_artifact_age_seconds: int = 86_400
    max_future_skew_seconds: int = 60
    max_artifact_bytes: int = 2_097_152
    max_cases: int = 500
    max_tokens_per_run: int = 4_096
    max_input_bytes: int = 16_384
    max_json_nodes: int = 100_000
    max_json_depth: int = 16
    max_reported_findings: int = 100

    def validate(self) -> None:
        unit = (
            "min_signed_cosine",
            "min_top_k_jaccard",
            "max_failed_case_fraction",
            "top_k_fraction",
        )
        for name in unit:
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"policy {name} must be numeric")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"policy {name} must be within [0, 1]")
        positive_float = (
            "max_relative_l1",
            "max_output_margin_drift",
            "max_attribution_sum_drift",
            "min_vector_l1",
        )
        for name in positive_float:
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"policy {name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"policy {name} must be finite and non-negative")
        positive_int = (
            "max_artifact_age_seconds",
            "max_future_skew_seconds",
            "max_artifact_bytes",
            "max_cases",
            "max_tokens_per_run",
            "max_input_bytes",
            "max_json_nodes",
            "max_json_depth",
            "max_reported_findings",
        )
        for name in positive_int:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"policy {name} must be a positive integer")


@dataclass(frozen=True)
class _Token:
    start: int
    end: int
    attribution: float


@dataclass(frozen=True)
class _Run:
    tokenizer_digest: str
    output_margin: float
    tokens: tuple[_Token, ...]


@dataclass(frozen=True)
class _Case:
    case_id: str
    input_digest: str
    input_byte_length: int
    reference: _Run
    candidate: _Run


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _private_ref(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _reject_constant(value: str) -> None:
    raise MalformedArtifact(f"non-finite JSON number: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MalformedArtifact(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_artifact(path: Path, policy: TokenizerMigrationPolicy) -> dict[str, Any]:
    """Read strict, resource-bounded JSON from disk."""

    try:
        size = path.stat().st_size
        if size > policy.max_artifact_bytes:
            raise MalformedArtifact("artifact exceeds byte budget")
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except MalformedArtifact:
        raise
    except (OSError, UnicodeDecodeError) as exc:
        raise MalformedArtifact("artifact must be readable UTF-8") from exc
    if len(raw) > policy.max_artifact_bytes:
        raise MalformedArtifact("artifact exceeds byte budget")
    try:
        artifact = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except MalformedArtifact:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise MalformedArtifact("artifact is not valid JSON") from exc
    if not isinstance(artifact, dict):
        raise MalformedArtifact("artifact root must be an object")
    _validate_tree_budget(artifact, policy)
    return artifact


def _validate_tree_budget(value: Any, policy: TokenizerMigrationPolicy) -> None:
    nodes = 0
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > policy.max_json_nodes:
            raise MalformedArtifact("artifact exceeds JSON node budget")
        if depth > policy.max_json_depth:
            raise MalformedArtifact("artifact exceeds JSON depth budget")
        if isinstance(item, dict):
            stack.extend((key, depth + 1) for key in item)
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise MalformedArtifact("artifact contains a non-finite number")


def _exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        raise MalformedArtifact(
            f"{context} fields mismatch; "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MalformedArtifact(f"{context} must be an object")
    return value


def _array(value: Any, context: str, maximum: int) -> list[Any]:
    if not isinstance(value, list):
        raise MalformedArtifact(f"{context} must be an array")
    if len(value) > maximum:
        raise MalformedArtifact(f"{context} exceeds item budget")
    return value


def _finite(value: Any, context: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise MalformedArtifact(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MalformedArtifact(f"{context} must be finite")
    return result


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise MalformedArtifact(f"{context} must be an integer >= {minimum}")
    return value


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise MalformedArtifact(f"{context} must be a lowercase SHA-256 digest")
    return value


def _timestamp(value: Any, context: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise MalformedArtifact(f"{context} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise MalformedArtifact(f"{context} is not a valid timestamp") from exc
    return parsed.astimezone(UTC)


def _parse_run(
    raw: Any,
    context: str,
    input_length: int,
    policy: TokenizerMigrationPolicy,
) -> _Run:
    value = _object(raw, context)
    _exact_keys(value, {"tokenizer_digest", "output_margin", "tokens"}, context)
    tokenizer_digest = _sha256(value["tokenizer_digest"], f"{context}.tokenizer_digest")
    output_margin = _finite(value["output_margin"], f"{context}.output_margin")
    tokens: list[_Token] = []
    for index, raw_token in enumerate(
        _array(value["tokens"], f"{context}.tokens", policy.max_tokens_per_run)
    ):
        token = _object(raw_token, f"{context}.tokens[{index}]")
        _exact_keys(token, {"start_byte", "end_byte", "attribution"}, "token")
        start = _integer(token["start_byte"], "token.start_byte")
        end = _integer(token["end_byte"], "token.end_byte", minimum=1)
        attribution = _finite(token["attribution"], "token.attribution")
        if start >= end or end > input_length:
            raise MalformedArtifact(f"{context} contains an invalid token span")
        tokens.append(_Token(start, end, attribution))
    if not tokens:
        raise MalformedArtifact(f"{context}.tokens cannot be empty")
    ordered = sorted(tokens, key=lambda token: (token.start, token.end))
    cursor = 0
    for token in ordered:
        if token.start != cursor:
            raise MalformedArtifact(f"{context} token spans must exactly cover the input")
        cursor = token.end
    if cursor != input_length:
        raise MalformedArtifact(f"{context} token spans must exactly cover the input")
    return _Run(tokenizer_digest, output_margin, tuple(ordered))


def _parse_artifact(
    artifact: dict[str, Any], policy: TokenizerMigrationPolicy
) -> tuple[datetime, tuple[_Case, ...]]:
    _exact_keys(
        artifact,
        {
            "schema_version",
            "created_at",
            "benchmark_digest",
            "model_weights_digest",
            "target_digest",
            "cases",
        },
        "artifact",
    )
    if artifact["schema_version"] != ARTIFACT_SCHEMA:
        raise MalformedArtifact("unsupported schema_version")
    created_at = _timestamp(artifact["created_at"], "created_at")
    _sha256(artifact["benchmark_digest"], "benchmark_digest")
    _sha256(artifact["model_weights_digest"], "model_weights_digest")
    _sha256(artifact["target_digest"], "target_digest")
    cases: list[_Case] = []
    seen_ids: set[str] = set()
    for index, raw_case in enumerate(_array(artifact["cases"], "cases", policy.max_cases)):
        value = _object(raw_case, f"cases[{index}]")
        _exact_keys(
            value,
            {
                "case_id",
                "input_digest",
                "input_byte_length",
                "reference",
                "candidate",
            },
            f"cases[{index}]",
        )
        case_id = value["case_id"]
        if not isinstance(case_id, str) or not _ID_RE.fullmatch(case_id):
            raise MalformedArtifact("case_id is invalid")
        if case_id in seen_ids:
            raise MalformedArtifact("duplicate case_id")
        seen_ids.add(case_id)
        input_digest = _sha256(value["input_digest"], "input_digest")
        input_length = _integer(value["input_byte_length"], "input_byte_length", minimum=1)
        if input_length > policy.max_input_bytes:
            raise MalformedArtifact("input exceeds byte projection budget")
        reference = _parse_run(value["reference"], "reference", input_length, policy)
        candidate = _parse_run(value["candidate"], "candidate", input_length, policy)
        if reference.tokenizer_digest == candidate.tokenizer_digest:
            raise MalformedArtifact("reference and candidate tokenizer digests must differ")
        cases.append(_Case(case_id, input_digest, input_length, reference, candidate))
    if not cases:
        raise MalformedArtifact("cases cannot be empty")
    return created_at, tuple(cases)


def _project(run: _Run, length: int) -> list[float]:
    projected = [0.0] * length
    for token in run.tokens:
        share = token.attribution / (token.end - token.start)
        for position in range(token.start, token.end):
            projected[position] = share
    return projected


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(-1.0, min(1.0, numerator / (left_norm * right_norm)))


def _top_k_indices(values: list[float], fraction: float) -> set[int]:
    count = max(1, math.ceil(len(values) * fraction))
    ranked = sorted(range(len(values)), key=lambda index: (-abs(values[index]), index))
    return set(ranked[:count])


def audit_tokenizer_migration(
    artifact: dict[str, Any],
    policy: TokenizerMigrationPolicy | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Evaluate a tokenizer-migration attribution artifact."""

    policy = policy or TokenizerMigrationPolicy()
    policy.validate()
    current_time = (now or datetime.now(UTC)).astimezone(UTC)
    created_at, cases = _parse_artifact(artifact, policy)
    findings: list[dict[str, str]] = []
    metrics: list[dict[str, Any]] = []
    release_rejected = False

    def add(code: str, case_id: str | None = None) -> None:
        finding = {"code": code}
        if case_id is not None:
            finding["case_ref"] = _private_ref(case_id)
        findings.append(finding)

    if (created_at - current_time).total_seconds() > policy.max_future_skew_seconds:
        add("ARTIFACT_FROM_FUTURE")
        release_rejected = True
    if (current_time - created_at).total_seconds() > policy.max_artifact_age_seconds:
        add("STALE_ARTIFACT")
        release_rejected = True

    failed_cases = 0
    for case in cases:
        reference = _project(case.reference, case.input_byte_length)
        candidate = _project(case.candidate, case.input_byte_length)
        reference_l1 = sum(abs(value) for value in reference)
        candidate_l1 = sum(abs(value) for value in candidate)
        case_codes: list[str] = []
        if reference_l1 < policy.min_vector_l1 or candidate_l1 < policy.min_vector_l1:
            case_codes.append("DEGENERATE_ATTRIBUTION")
            signed_cosine = 0.0
            relative_l1 = math.inf
            top_k_jaccard = 0.0
        else:
            signed_cosine = _cosine(reference, candidate)
            relative_l1 = sum(
                abs(left - right) for left, right in zip(reference, candidate, strict=True)
            ) / max(reference_l1, candidate_l1)
            reference_top = _top_k_indices(reference, policy.top_k_fraction)
            candidate_top = _top_k_indices(candidate, policy.top_k_fraction)
            top_k_jaccard = len(reference_top & candidate_top) / len(reference_top | candidate_top)
            if signed_cosine < policy.min_signed_cosine:
                case_codes.append("SIGNED_COSINE_BELOW_MINIMUM")
            if relative_l1 > policy.max_relative_l1:
                case_codes.append("RELATIVE_L1_ABOVE_MAXIMUM")
            if top_k_jaccard < policy.min_top_k_jaccard:
                case_codes.append("TOP_K_OVERLAP_BELOW_MINIMUM")

        margin_drift = abs(case.reference.output_margin - case.candidate.output_margin)
        attribution_sum_drift = abs(sum(reference) - sum(candidate))
        if margin_drift > policy.max_output_margin_drift:
            case_codes.append("OUTPUT_MARGIN_DRIFT")
        if attribution_sum_drift > policy.max_attribution_sum_drift:
            case_codes.append("ATTRIBUTION_SUM_DRIFT")
        if case_codes:
            failed_cases += 1
            for code in sorted(set(case_codes)):
                add(code, case.case_id)
        metrics.append(
            {
                "case_ref": _private_ref(case.case_id),
                "input_byte_length": case.input_byte_length,
                "reference_token_count": len(case.reference.tokens),
                "candidate_token_count": len(case.candidate.tokens),
                "signed_cosine": signed_cosine if math.isfinite(signed_cosine) else None,
                "relative_l1": relative_l1 if math.isfinite(relative_l1) else None,
                "top_k_jaccard": top_k_jaccard,
                "output_margin_drift": margin_drift,
                "attribution_sum_drift": attribution_sum_drift,
            }
        )

    failed_fraction = failed_cases / len(cases)
    if failed_fraction > policy.max_failed_case_fraction:
        add("FAILED_CASE_FRACTION_EXCEEDED")
        release_rejected = True
    findings.sort(key=lambda item: (item["code"], item.get("case_ref", "")))
    metrics.sort(key=lambda item: item["case_ref"])
    policy_payload = asdict(policy)
    reported = findings[: policy.max_reported_findings]
    return {
        "schema_version": REPORT_SCHEMA,
        "accepted": not release_rejected,
        "reason_codes": sorted({finding["code"] for finding in findings}),
        "finding_count": len(findings),
        "reported_findings": reported,
        "truncated_findings": len(findings) - len(reported),
        "summary": {
            "case_count": len(cases),
            "failed_case_count": failed_cases,
            "failed_case_fraction": failed_fraction,
        },
        "case_metrics": metrics,
        "artifact_digest": _digest(artifact),
        "policy_digest": _digest(policy_payload),
        "evidence_id": _digest({"artifact": artifact, "policy": policy_payload}),
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            os.chmod(temporary, 0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-signed-cosine", type=float, default=0.90)
    parser.add_argument("--min-top-k-jaccard", type=float, default=0.70)
    parser.add_argument("--max-relative-l1", type=float, default=0.30)
    parser.add_argument("--max-output-margin-drift", type=float, default=1e-4)
    parser.add_argument("--max-attribution-sum-drift", type=float, default=1e-5)
    parser.add_argument("--max-failed-case-fraction", type=float, default=0.0)
    parser.add_argument("--max-artifact-age-seconds", type=int, default=86_400)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        policy = TokenizerMigrationPolicy(
            min_signed_cosine=args.min_signed_cosine,
            min_top_k_jaccard=args.min_top_k_jaccard,
            max_relative_l1=args.max_relative_l1,
            max_output_margin_drift=args.max_output_margin_drift,
            max_attribution_sum_drift=args.max_attribution_sum_drift,
            max_failed_case_fraction=args.max_failed_case_fraction,
            max_artifact_age_seconds=args.max_artifact_age_seconds,
        )
        policy.validate()
        artifact = load_artifact(args.artifact, policy)
        report = audit_tokenizer_migration(artifact, policy)
        payload = _canonical_bytes(report) + b"\n"
        if args.output:
            _atomic_write(args.output, payload)
        else:
            sys.stdout.buffer.write(payload)
        return 0 if report["accepted"] else 2
    except (MalformedArtifact, ValueError) as exc:
        error = {
            "schema_version": REPORT_SCHEMA,
            "accepted": False,
            "error": "malformed_artifact",
            "detail": str(exc),
        }
        sys.stderr.buffer.write(_canonical_bytes(error) + b"\n")
        return 3
    except OSError:
        error = {
            "schema_version": REPORT_SCHEMA,
            "accepted": False,
            "error": "io_error",
        }
        sys.stderr.buffer.write(_canonical_bytes(error) + b"\n")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
