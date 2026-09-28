from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

MAX_ARTIFACT_BYTES = 2 * 1024 * 1024
MAX_CASES = 1_024
MAX_FEATURES = 8_192
MAX_TOTAL_VALUES = 2_000_000
MAX_FINDINGS = 256
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,127}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class ConvergenceFormatError(ValueError):
    """The convergence artifact is malformed or exceeds a resource budget."""


@dataclass(frozen=True, slots=True)
class AttributionEstimate:
    integration_steps: int
    feature_ids: tuple[str, ...]
    attributions: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ConvergenceCase:
    case_id: str
    output_delta: float
    estimates: tuple[AttributionEstimate, ...]


@dataclass(frozen=True, slots=True)
class ConvergenceArtifact:
    schema_version: int
    benchmark_id: str
    model_digest: str
    method_digest: str
    created_at: datetime
    cases: tuple[ConvergenceCase, ...]


@dataclass(frozen=True, slots=True)
class ConvergenceBinding:
    benchmark_id: str
    model_digest: str
    method_digest: str

    def __post_init__(self) -> None:
        _identifier(self.benchmark_id, "benchmark_id")
        _digest(self.model_digest, "model_digest")
        _digest(self.method_digest, "method_digest")


@dataclass(frozen=True, slots=True)
class ConvergencePolicy:
    required_steps: tuple[int, ...] = (8, 16, 32, 64)
    max_final_completeness_error: float = 0.02
    max_final_relative_l1_change: float = 0.05
    min_final_cosine: float = 0.995
    max_final_to_initial_residual_ratio: float = 1.0
    min_abs_output_delta: float = 1e-6
    max_failed_case_fraction: float = 0.0
    max_age_seconds: int = 86_400
    max_future_skew_seconds: int = 30

    def __post_init__(self) -> None:
        if len(self.required_steps) < 2 or tuple(sorted(set(self.required_steps))) != self.required_steps:
            raise ValueError("required_steps must contain at least two unique increasing values")
        if any(isinstance(step, bool) or not 1 <= step <= 65_536 for step in self.required_steps):
            raise ValueError("required_steps are outside the accepted range")
        for name, value in (
            ("max_final_completeness_error", self.max_final_completeness_error),
            ("max_final_relative_l1_change", self.max_final_relative_l1_change),
            ("max_failed_case_fraction", self.max_failed_case_fraction),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        if not math.isfinite(self.min_final_cosine) or not -1.0 <= self.min_final_cosine <= 1.0:
            raise ValueError("min_final_cosine must be finite and between -1 and 1")
        if (
            not math.isfinite(self.max_final_to_initial_residual_ratio)
            or self.max_final_to_initial_residual_ratio < 0
        ):
            raise ValueError("max_final_to_initial_residual_ratio must be finite and non-negative")
        if not math.isfinite(self.min_abs_output_delta) or self.min_abs_output_delta <= 0:
            raise ValueError("min_abs_output_delta must be finite and positive")
        if not 1 <= self.max_age_seconds <= 31_536_000:
            raise ValueError("max_age_seconds is outside the accepted range")
        if not 0 <= self.max_future_skew_seconds <= 300:
            raise ValueError("max_future_skew_seconds is outside the accepted range")


@dataclass(frozen=True, slots=True)
class CaseConvergence:
    case_hash: str
    accepted: bool
    finding_codes: tuple[str, ...]
    feature_count: int
    final_completeness_error: float
    final_relative_l1_change: float
    final_cosine: float
    final_to_initial_residual_ratio: float


@dataclass(frozen=True, slots=True)
class ConvergenceReport:
    accepted: bool
    finding_codes: tuple[str, ...]
    artifact_digest: str
    policy_digest: str
    case_count: int
    failed_case_count: int
    failed_case_fraction: float
    cases: tuple[CaseConvergence, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConvergenceFormatError("duplicate JSON field")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ConvergenceFormatError(f"non-finite number {value!r} is not allowed")


def _object(value: Any, fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ConvergenceFormatError(f"{name} has missing or unknown fields")
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ConvergenceFormatError(f"{name} is invalid")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ConvergenceFormatError(f"{name} must be a canonical SHA-256 digest")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConvergenceFormatError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ConvergenceFormatError(f"{name} must be finite")
    return result


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or len(value) > 64:
        raise ConvergenceFormatError("created_at is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ConvergenceFormatError("created_at is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ConvergenceFormatError("created_at must include a UTC offset")
    return parsed.astimezone(UTC)


_ARTIFACT_FIELDS = frozenset(
    {"schema_version", "benchmark_id", "model_digest", "method_digest", "created_at", "cases"}
)
_CASE_FIELDS = frozenset({"case_id", "output_delta", "estimates"})
_ESTIMATE_FIELDS = frozenset({"integration_steps", "feature_ids", "attributions"})


def parse_artifact(raw: bytes) -> ConvergenceArtifact:
    if not raw or len(raw) > MAX_ARTIFACT_BYTES:
        raise ConvergenceFormatError("artifact byte size is outside the accepted range")
    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConvergenceFormatError("artifact is not valid UTF-8 JSON") from exc
    document = _object(decoded, _ARTIFACT_FIELDS, "artifact")
    if document["schema_version"] != 1:
        raise ConvergenceFormatError("unsupported schema_version")
    raw_cases = document["cases"]
    if not isinstance(raw_cases, list) or not 1 <= len(raw_cases) <= MAX_CASES:
        raise ConvergenceFormatError("case count is outside the accepted range")
    cases: list[ConvergenceCase] = []
    total_values = 0
    for raw_case in raw_cases:
        case = _object(raw_case, _CASE_FIELDS, "case")
        raw_estimates = case["estimates"]
        if not isinstance(raw_estimates, list) or not 2 <= len(raw_estimates) <= 32:
            raise ConvergenceFormatError("estimate count is outside the accepted range")
        estimates: list[AttributionEstimate] = []
        for raw_estimate in raw_estimates:
            estimate = _object(raw_estimate, _ESTIMATE_FIELDS, "estimate")
            steps = estimate["integration_steps"]
            if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 65_536:
                raise ConvergenceFormatError("integration_steps is outside the accepted range")
            feature_ids = estimate["feature_ids"]
            attributions = estimate["attributions"]
            if (
                not isinstance(feature_ids, list)
                or not isinstance(attributions, list)
                or not 1 <= len(feature_ids) <= MAX_FEATURES
                or len(feature_ids) != len(attributions)
            ):
                raise ConvergenceFormatError("feature vectors are malformed or outside budget")
            parsed_ids = tuple(_identifier(item, "feature_id") for item in feature_ids)
            if len(set(parsed_ids)) != len(parsed_ids):
                raise ConvergenceFormatError("feature IDs must be unique")
            parsed_values = tuple(_finite(item, "attribution") for item in attributions)
            total_values += len(parsed_values)
            if total_values > MAX_TOTAL_VALUES:
                raise ConvergenceFormatError("total attribution value budget exceeded")
            estimates.append(AttributionEstimate(steps, parsed_ids, parsed_values))
        cases.append(
            ConvergenceCase(
                case_id=_identifier(case["case_id"], "case_id"),
                output_delta=_finite(case["output_delta"], "output_delta"),
                estimates=tuple(estimates),
            )
        )
    if len({case.case_id for case in cases}) != len(cases):
        raise ConvergenceFormatError("case IDs must be unique")
    return ConvergenceArtifact(
        schema_version=1,
        benchmark_id=_identifier(document["benchmark_id"], "benchmark_id"),
        model_digest=_digest(document["model_digest"], "model_digest"),
        method_digest=_digest(document["method_digest"], "method_digest"),
        created_at=_timestamp(document["created_at"]),
        cases=tuple(cases),
    )


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0 if left == right else 0.0
    value = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    return min(1.0, max(-1.0, value))


def _canonical_artifact(artifact: ConvergenceArtifact) -> bytes:
    payload = {
        "benchmark_id": artifact.benchmark_id,
        "cases": [
            {
                "case_id": case.case_id,
                "estimates": [
                    asdict(estimate)
                    for estimate in sorted(case.estimates, key=lambda row: row.integration_steps)
                ],
                "output_delta": case.output_delta,
            }
            for case in sorted(artifact.cases, key=lambda row: row.case_id)
        ],
        "created_at": artifact.created_at.isoformat(),
        "method_digest": artifact.method_digest,
        "model_digest": artifact.model_digest,
        "schema_version": 1,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _policy_digest(policy: ConvergencePolicy, binding: ConvergenceBinding) -> str:
    payload = {"binding": asdict(binding), "policy": asdict(policy)}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def audit_convergence(
    artifact: ConvergenceArtifact,
    binding: ConvergenceBinding,
    policy: ConvergencePolicy,
    *,
    now: datetime,
) -> ConvergenceReport:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    now = now.astimezone(UTC)
    report_findings: set[str] = set()
    if artifact.benchmark_id != binding.benchmark_id:
        report_findings.add("benchmark_binding_mismatch")
    if artifact.model_digest != binding.model_digest:
        report_findings.add("model_binding_mismatch")
    if artifact.method_digest != binding.method_digest:
        report_findings.add("method_binding_mismatch")
    if artifact.created_at > now + timedelta(seconds=policy.max_future_skew_seconds):
        report_findings.add("artifact_future_dated")
    if now - artifact.created_at > timedelta(seconds=policy.max_age_seconds):
        report_findings.add("artifact_stale")

    rows: list[CaseConvergence] = []
    for case in sorted(artifact.cases, key=lambda item: item.case_id):
        findings: set[str] = set()
        estimates = sorted(case.estimates, key=lambda item: item.integration_steps)
        steps = tuple(item.integration_steps for item in estimates)
        if steps != policy.required_steps:
            findings.add("step_schedule_mismatch")
        feature_ids = estimates[0].feature_ids
        if any(item.feature_ids != feature_ids for item in estimates[1:]):
            findings.add("feature_alignment_mismatch")
        if abs(case.output_delta) < policy.min_abs_output_delta:
            findings.add("output_delta_too_small")

        residuals = [abs(sum(item.attributions) - case.output_delta) for item in estimates]
        denominator = max(abs(case.output_delta), policy.min_abs_output_delta)
        completeness = residuals[-1] / denominator
        previous = estimates[-2].attributions
        final = estimates[-1].attributions
        relative_l1 = sum(abs(a - b) for a, b in zip(previous, final, strict=True)) / max(
            sum(abs(value) for value in final), policy.min_abs_output_delta
        )
        cosine = _cosine(previous, final)
        residual_ratio = residuals[-1] / max(residuals[0], policy.min_abs_output_delta)
        if completeness > policy.max_final_completeness_error:
            findings.add("completeness_not_converged")
        if relative_l1 > policy.max_final_relative_l1_change:
            findings.add("attribution_l1_not_converged")
        if cosine < policy.min_final_cosine:
            findings.add("attribution_direction_not_converged")
        if residual_ratio > policy.max_final_to_initial_residual_ratio:
            findings.add("completeness_did_not_improve")
        ordered = tuple(sorted(findings))
        rows.append(
            CaseConvergence(
                case_hash=hashlib.sha256(case.case_id.encode()).hexdigest()[:16],
                accepted=not ordered,
                finding_codes=ordered,
                feature_count=len(feature_ids),
                final_completeness_error=completeness,
                final_relative_l1_change=relative_l1,
                final_cosine=cosine,
                final_to_initial_residual_ratio=residual_ratio,
            )
        )
    failed = sum(not row.accepted for row in rows)
    fraction = failed / len(rows)
    if fraction > policy.max_failed_case_fraction:
        report_findings.add("failed_case_fraction_exceeded")
    bounded_findings = tuple(sorted(report_findings))[:MAX_FINDINGS]
    return ConvergenceReport(
        accepted=not bounded_findings,
        finding_codes=bounded_findings,
        artifact_digest=hashlib.sha256(_canonical_artifact(artifact)).hexdigest(),
        policy_digest=_policy_digest(policy, binding),
        case_count=len(rows),
        failed_case_count=failed,
        failed_case_fraction=fraction,
        cases=tuple(rows),
    )
