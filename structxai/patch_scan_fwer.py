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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


class ArtifactMalformed(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ScanPolicy:
    max_input_bytes: int = 4_194_304
    max_cases: int = 256
    max_sites: int = 512
    max_null_replicates: int = 2_000
    max_total_patch_values: int = 2_000_000
    min_cases: int = 3
    min_sites: int = 4
    min_null_replicates: int = 19
    family_wise_alpha: float = 0.05
    min_recovery: float = 0.20
    max_recovery: float = 1.50
    min_abs_corruption_gap: float = 0.10
    min_significant_case_fraction: float = 0.60
    min_recurrent_site_fraction: float = 0.40
    max_age_seconds: int = 86_400
    max_future_skew_seconds: int = 60
    max_findings: int = 256

    def __post_init__(self) -> None:
        checks = (
            (1 <= self.max_input_bytes <= 33_554_432, "max_input_bytes"),
            (1 <= self.max_cases <= 10_000, "max_cases"),
            (1 <= self.max_sites <= 10_000, "max_sites"),
            (1 <= self.max_null_replicates <= 100_000, "max_null_replicates"),
            (1 <= self.max_total_patch_values <= 50_000_000, "max_total_patch_values"),
            (1 <= self.min_cases <= self.max_cases, "min_cases"),
            (1 <= self.min_sites <= self.max_sites, "min_sites"),
            (
                1 <= self.min_null_replicates <= self.max_null_replicates,
                "min_null_replicates",
            ),
            (0.0 < self.family_wise_alpha < 1.0, "family_wise_alpha"),
            (0.0 <= self.min_recovery <= self.max_recovery, "recovery_range"),
            (self.min_abs_corruption_gap > 0.0, "min_abs_corruption_gap"),
            (
                0.0 <= self.min_significant_case_fraction <= 1.0,
                "min_significant_case_fraction",
            ),
            (
                0.0 <= self.min_recurrent_site_fraction <= 1.0,
                "min_recurrent_site_fraction",
            ),
            (1 <= self.max_age_seconds <= 604_800, "max_age_seconds"),
            (0 <= self.max_future_skew_seconds <= 3_600, "max_future_skew_seconds"),
            (1 <= self.max_findings <= 10_000, "max_findings"),
        )
        for valid, name in checks:
            if not valid:
                raise ValueError(f"{name} outside supported range")
        if 1.0 / (self.min_null_replicates + 1) > self.family_wise_alpha:
            raise ValueError("null replicate count cannot resolve family_wise_alpha")


@dataclass(frozen=True)
class Finding:
    scope: str
    subject_sha256: str
    code: str


@dataclass(frozen=True)
class CaseSummary:
    case_sha256: str
    significant_site_count: int
    minimum_corrected_p: float | None
    peak_recovery: float | None
    passed: bool


@dataclass(frozen=True)
class RecurrentSite:
    site_sha256: str
    significant_case_count: int
    significant_case_fraction: float


@dataclass(frozen=True)
class PatchScanReport:
    schema_version: int
    status: str
    accepted: bool
    artifact_sha256: str
    policy_sha256: str
    case_count: int
    site_count: int
    null_replicates_per_case: int
    total_patch_values: int
    significant_case_count: int
    significant_case_fraction: float
    recurrent_site_count: int
    finding_count: int
    findings_truncated: bool
    cases: tuple[CaseSummary, ...]
    recurrent_sites: tuple[RecurrentSite, ...]
    findings: tuple[Finding, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["cases"] = [asdict(item) for item in self.cases]
        result["recurrent_sites"] = [asdict(item) for item in self.recurrent_sites]
        result["findings"] = [asdict(item) for item in self.findings]
        return result


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactMalformed("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ArtifactMalformed("NON_FINITE_NUMBER")


def load_artifact(raw: bytes, policy: ScanPolicy | None = None) -> dict[str, Any]:
    selected_policy = policy or ScanPolicy()
    if len(raw) > selected_policy.max_input_bytes:
        raise ArtifactMalformed("INPUT_TOO_LARGE")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactMalformed("INVALID_UTF8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except ArtifactMalformed:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ArtifactMalformed("INVALID_JSON") from exc
    if not isinstance(value, dict):
        raise ArtifactMalformed("ROOT_NOT_OBJECT")
    return value


def audit_patch_scan(
    artifact: dict[str, Any],
    *,
    as_of: datetime,
    policy: ScanPolicy | None = None,
) -> PatchScanReport:
    selected_policy = policy or ScanPolicy()
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    as_of = as_of.astimezone(UTC)
    _expect_keys(
        artifact,
        {
            "schema_version",
            "experiment_id",
            "model_sha256",
            "patch_config_sha256",
            "eligible_sites_sha256",
            "created_at",
            "cases",
        },
        "ROOT_FIELDS",
    )
    if artifact["schema_version"] != 1 or isinstance(artifact["schema_version"], bool):
        raise ArtifactMalformed("SCHEMA_VERSION")
    _identifier(artifact["experiment_id"], "EXPERIMENT_ID")
    _digest(artifact["model_sha256"], "MODEL_SHA256")
    _digest(artifact["patch_config_sha256"], "PATCH_CONFIG_SHA256")
    _digest(artifact["eligible_sites_sha256"], "ELIGIBLE_SITES_SHA256")
    created_at = _timestamp(artifact["created_at"], "CREATED_AT")
    if created_at > as_of + timedelta(seconds=selected_policy.max_future_skew_seconds):
        raise ArtifactMalformed("EVIDENCE_FROM_FUTURE")
    if as_of - created_at > timedelta(seconds=selected_policy.max_age_seconds):
        raise ArtifactMalformed("STALE_EVIDENCE")

    raw_cases = artifact["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ArtifactMalformed("CASES_REQUIRED")
    if len(raw_cases) > selected_policy.max_cases:
        raise ArtifactMalformed("CASE_BUDGET_EXCEEDED")

    seen_cases: set[str] = set()
    expected_sites: tuple[str, ...] | None = None
    expected_replicates: int | None = None
    total_values = 0
    summaries: list[CaseSummary] = []
    findings: list[Finding] = []
    significant_support: dict[str, int] = {}
    canonical_cases: list[dict[str, Any]] = []

    for raw_case in raw_cases:
        parsed = _parse_case(
            raw_case,
            policy=selected_policy,
            seen_cases=seen_cases,
        )
        site_ids = tuple(sorted(parsed["observed"]))
        if expected_sites is None:
            expected_sites = site_ids
            significant_support = dict.fromkeys(site_ids, 0)
        elif site_ids != expected_sites:
            raise ArtifactMalformed("CASE_SITE_SET_MISMATCH")
        replicate_count = len(parsed["null_replicates"])
        if expected_replicates is None:
            expected_replicates = replicate_count
        elif replicate_count != expected_replicates:
            raise ArtifactMalformed("NULL_REPLICATE_COUNT_MISMATCH")
        total_values += len(site_ids) * (replicate_count + 1)
        if total_values > selected_policy.max_total_patch_values:
            raise ArtifactMalformed("PATCH_VALUE_BUDGET_EXCEEDED")

        summary, case_findings, significant_sites = _evaluate_case(parsed, selected_policy)
        summaries.append(summary)
        findings.extend(case_findings)
        for site_id in significant_sites:
            significant_support[site_id] += 1
        canonical_cases.append(parsed["canonical"])

    assert expected_sites is not None
    assert expected_replicates is not None
    expected_sites_digest = _sha256(_canonical_json(list(expected_sites)))
    if artifact["eligible_sites_sha256"] != expected_sites_digest:
        raise ArtifactMalformed("ELIGIBLE_SITES_DIGEST_MISMATCH")
    aggregate_hash = _sha256(artifact["experiment_id"].encode())
    if len(raw_cases) < selected_policy.min_cases:
        findings.append(Finding("aggregate", aggregate_hash, "CASE_EVIDENCE_UNDERPOWERED"))
    if len(expected_sites) < selected_policy.min_sites:
        findings.append(Finding("aggregate", aggregate_hash, "SITE_EVIDENCE_UNDERPOWERED"))
    if expected_replicates < selected_policy.min_null_replicates:
        findings.append(Finding("aggregate", aggregate_hash, "NULL_EVIDENCE_UNDERPOWERED"))

    significant_case_count = sum(summary.passed for summary in summaries)
    significant_case_fraction = significant_case_count / len(summaries)
    if significant_case_fraction < selected_policy.min_significant_case_fraction:
        findings.append(Finding("aggregate", aggregate_hash, "SIGNIFICANT_CASE_FRACTION_BELOW_POLICY"))

    recurrent_sites = tuple(
        sorted(
            (
                RecurrentSite(
                    site_sha256=_sha256(site_id.encode()),
                    significant_case_count=count,
                    significant_case_fraction=count / len(summaries),
                )
                for site_id, count in significant_support.items()
                if count / len(summaries) >= selected_policy.min_recurrent_site_fraction
            ),
            key=lambda item: (-item.significant_case_count, item.site_sha256),
        )
    )
    if not recurrent_sites:
        findings.append(Finding("aggregate", aggregate_hash, "NO_RECURRENT_SIGNIFICANT_SITE"))

    findings.sort(key=lambda item: (item.scope, item.subject_sha256, item.code))
    truncated = len(findings) > selected_policy.max_findings
    canonical_artifact = {
        key: artifact[key]
        for key in (
            "schema_version",
            "experiment_id",
            "model_sha256",
            "patch_config_sha256",
            "eligible_sites_sha256",
            "created_at",
        )
    }
    canonical_artifact["cases"] = sorted(
        canonical_cases,
        key=lambda item: item["case_id"],
    )
    accepted = not findings
    return PatchScanReport(
        schema_version=1,
        status="accepted" if accepted else "policy_rejected",
        accepted=accepted,
        artifact_sha256=_sha256(_canonical_json(canonical_artifact)),
        policy_sha256=_sha256(_canonical_json(asdict(selected_policy))),
        case_count=len(summaries),
        site_count=len(expected_sites),
        null_replicates_per_case=expected_replicates,
        total_patch_values=total_values,
        significant_case_count=significant_case_count,
        significant_case_fraction=significant_case_fraction,
        recurrent_site_count=len(recurrent_sites),
        finding_count=len(findings),
        findings_truncated=truncated,
        cases=tuple(sorted(summaries, key=lambda item: item.case_sha256)),
        recurrent_sites=recurrent_sites,
        findings=tuple(findings[: selected_policy.max_findings]),
    )


def _parse_case(
    raw_case: Any,
    *,
    policy: ScanPolicy,
    seen_cases: set[str],
) -> dict[str, Any]:
    if not isinstance(raw_case, dict):
        raise ArtifactMalformed("CASE_NOT_OBJECT")
    _expect_keys(
        raw_case,
        {"case_id", "clean_margin", "corrupted_margin", "sites", "null_replicates"},
        "CASE_FIELDS",
    )
    case_id = _identifier(raw_case["case_id"], "CASE_ID")
    if case_id in seen_cases:
        raise ArtifactMalformed("DUPLICATE_CASE_ID")
    seen_cases.add(case_id)
    clean_margin = _number(raw_case["clean_margin"], "CLEAN_MARGIN")
    corrupted_margin = _number(raw_case["corrupted_margin"], "CORRUPTED_MARGIN")
    observed = _site_values(raw_case["sites"], policy, "SITE")
    null_replicates = raw_case["null_replicates"]
    if not isinstance(null_replicates, list) or not null_replicates:
        raise ArtifactMalformed("NULL_REPLICATES_REQUIRED")
    if len(null_replicates) > policy.max_null_replicates:
        raise ArtifactMalformed("NULL_REPLICATE_BUDGET_EXCEEDED")

    seen_replicates: set[str] = set()
    parsed_nulls: list[tuple[str, dict[str, float]]] = []
    expected_sites = set(observed)
    for replicate in null_replicates:
        if not isinstance(replicate, dict):
            raise ArtifactMalformed("NULL_REPLICATE_NOT_OBJECT")
        _expect_keys(replicate, {"replicate_id", "sites"}, "NULL_REPLICATE_FIELDS")
        replicate_id = _identifier(replicate["replicate_id"], "REPLICATE_ID")
        if replicate_id in seen_replicates:
            raise ArtifactMalformed("DUPLICATE_REPLICATE_ID")
        seen_replicates.add(replicate_id)
        values = _site_values(replicate["sites"], policy, "NULL_SITE")
        if set(values) != expected_sites:
            raise ArtifactMalformed("NULL_SITE_SET_MISMATCH")
        parsed_nulls.append((replicate_id, values))

    canonical = {
        "case_id": case_id,
        "clean_margin": clean_margin,
        "corrupted_margin": corrupted_margin,
        "sites": [{"site_id": site_id, "patched_margin": observed[site_id]} for site_id in sorted(observed)],
        "null_replicates": [
            {
                "replicate_id": replicate_id,
                "sites": [
                    {"site_id": site_id, "patched_margin": values[site_id]} for site_id in sorted(values)
                ],
            }
            for replicate_id, values in sorted(parsed_nulls)
        ],
    }
    return {
        "case_id": case_id,
        "clean_margin": clean_margin,
        "corrupted_margin": corrupted_margin,
        "observed": observed,
        "null_replicates": tuple(values for _, values in parsed_nulls),
        "canonical": canonical,
    }


def _site_values(value: Any, policy: ScanPolicy, prefix: str) -> dict[str, float]:
    if not isinstance(value, list) or not value:
        raise ArtifactMalformed(f"{prefix}_VALUES_REQUIRED")
    if len(value) > policy.max_sites:
        raise ArtifactMalformed("SITE_BUDGET_EXCEEDED")
    result: dict[str, float] = {}
    for item in value:
        if not isinstance(item, dict):
            raise ArtifactMalformed(f"{prefix}_NOT_OBJECT")
        _expect_keys(item, {"site_id", "patched_margin"}, f"{prefix}_FIELDS")
        site_id = _identifier(item["site_id"], "SITE_ID")
        if site_id in result:
            raise ArtifactMalformed(f"DUPLICATE_{prefix}_ID")
        result[site_id] = _number(item["patched_margin"], "PATCHED_MARGIN")
    return result


def _evaluate_case(
    parsed: dict[str, Any],
    policy: ScanPolicy,
) -> tuple[CaseSummary, list[Finding], set[str]]:
    case_id = parsed["case_id"]
    case_hash = _sha256(case_id.encode())
    gap = parsed["clean_margin"] - parsed["corrupted_margin"]
    if abs(gap) < policy.min_abs_corruption_gap:
        return (
            CaseSummary(case_hash, 0, None, None, False),
            [Finding("case", case_hash, "CORRUPTION_GAP_BELOW_POLICY")],
            set(),
        )

    observed_recoveries = {
        site_id: (margin - parsed["corrupted_margin"]) / gap for site_id, margin in parsed["observed"].items()
    }
    null_maxima = [
        max((margin - parsed["corrupted_margin"]) / gap for margin in values.values())
        for values in parsed["null_replicates"]
    ]
    corrected_p = {
        site_id: (1 + sum(null_maximum >= recovery for null_maximum in null_maxima)) / (len(null_maxima) + 1)
        for site_id, recovery in observed_recoveries.items()
    }
    significant = {
        site_id
        for site_id, recovery in observed_recoveries.items()
        if policy.min_recovery <= recovery <= policy.max_recovery
        and corrected_p[site_id] <= policy.family_wise_alpha
    }
    findings: list[Finding] = []
    if not significant:
        findings.append(Finding("case", case_hash, "NO_FWER_SIGNIFICANT_SITE"))
    if max(observed_recoveries.values()) > policy.max_recovery:
        findings.append(Finding("case", case_hash, "PATCH_RECOVERY_OVERSHOOT"))
    return (
        CaseSummary(
            case_sha256=case_hash,
            significant_site_count=len(significant),
            minimum_corrected_p=min(corrected_p.values()),
            peak_recovery=max(observed_recoveries.values()),
            passed=bool(significant),
        ),
        findings,
        significant,
    )


def _expect_keys(value: dict[str, Any], expected: set[str], code: str) -> None:
    if set(value) != expected:
        raise ArtifactMalformed(code)


def _identifier(value: Any, code: str) -> str:
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        raise ArtifactMalformed(code)
    return value


def _digest(value: Any, code: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ArtifactMalformed(code)
    return value


def _number(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactMalformed(code)
    result = float(value)
    if not math.isfinite(result) or abs(result) > 1_000_000.0:
        raise ArtifactMalformed(code)
    return result


def _timestamp(value: Any, code: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ArtifactMalformed(code)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ArtifactMalformed(code) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ArtifactMalformed(code)
    return parsed.astimezone(UTC)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit activation-patching site scans with max-statistic FWER control"
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--as-of", help="UTC RFC3339 timestamp; defaults to current UTC")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    policy = ScanPolicy()
    try:
        artifact = load_artifact(args.artifact.read_bytes(), policy)
        as_of = _timestamp(args.as_of, "AS_OF") if args.as_of else datetime.now(UTC)
        report = audit_patch_scan(artifact, as_of=as_of, policy=policy)
        payload = report.to_dict()
        exit_code = 0 if report.accepted else 2
    except (ArtifactMalformed, OSError) as exc:
        error = exc.code if isinstance(exc, ArtifactMalformed) else "ARTIFACT_IO_ERROR"
        payload = {"accepted": False, "error": error, "status": "malformed"}
        exit_code = 3
    if args.output:
        try:
            _write_json(args.output, payload)
        except OSError:
            return 3
    else:
        json.dump(payload, sys.stdout, sort_keys=True, separators=(",", ":"))
        sys.stdout.write("\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
