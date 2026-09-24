from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any


class PatchControlArtifactError(ValueError):
    def __init__(self, code: str, path: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.path = path


@dataclass(frozen=True)
class PatchObservation:
    site_id: str
    patched_margin: float


@dataclass(frozen=True)
class PatchControlCase:
    case_id: str
    clean_margin: float
    corrupted_margin: float
    target_patch: PatchObservation
    sham_patch: PatchObservation
    random_controls: tuple[PatchObservation, ...]


@dataclass(frozen=True)
class PatchControlPolicy:
    min_cases: int = 3
    min_random_controls: int = 5
    min_corruption_effect: float = 0.5
    min_target_recovery_fraction: float = 0.5
    max_target_recovery_fraction: float = 1.5
    max_sham_effect_fraction: float = 0.15
    max_control_p95_effect_fraction: float = 0.3
    min_specificity_gap: float = 0.25
    min_case_pass_rate: float = 0.8

    def __post_init__(self) -> None:
        for name in ("min_cases", "min_random_controls"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        numeric_fields = (
            "min_corruption_effect",
            "min_target_recovery_fraction",
            "max_target_recovery_fraction",
            "max_sham_effect_fraction",
            "max_control_p95_effect_fraction",
            "min_specificity_gap",
            "min_case_pass_rate",
        )
        for name in numeric_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.min_corruption_effect <= 0:
            raise ValueError("min_corruption_effect must be positive")
        if self.min_target_recovery_fraction < 0:
            raise ValueError("min_target_recovery_fraction cannot be negative")
        if self.max_target_recovery_fraction < self.min_target_recovery_fraction:
            raise ValueError("maximum target recovery cannot be below the minimum")
        if self.max_sham_effect_fraction < 0 or self.max_control_p95_effect_fraction < 0:
            raise ValueError("control-effect limits cannot be negative")
        if self.min_specificity_gap < 0:
            raise ValueError("min_specificity_gap cannot be negative")
        if not 0 <= self.min_case_pass_rate <= 1:
            raise ValueError("min_case_pass_rate must be in [0, 1]")


@dataclass(frozen=True)
class PatchControlCaseReport:
    case_id: str
    target_site_id: str
    sham_site_id: str
    corruption_effect: float
    target_recovery_fraction: float
    sham_effect_fraction: float
    control_p95_effect_fraction: float
    specificity_gap: float
    passed: bool
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class PatchControlReport:
    accepted: bool
    malformed: bool
    experiment_id: str | None
    reason_codes: tuple[str, ...]
    case_count: int
    passing_cases: int
    case_pass_rate: float
    mean_target_recovery_fraction: float | None
    median_target_recovery_fraction: float | None
    mean_specificity_gap: float | None
    cases: tuple[PatchControlCaseReport, ...]
    error_path: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _identifier(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise PatchControlArtifactError(
            "INVALID_IDENTIFIER", path, "identifier must contain 1-200 characters"
        )
    if any(ord(character) < 32 for character in value):
        raise PatchControlArtifactError(
            "INVALID_IDENTIFIER", path, "identifier must not contain control characters"
        )
    return value


def _finite(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PatchControlArtifactError("INVALID_MARGIN", path, "margin must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise PatchControlArtifactError("INVALID_MARGIN", path, "margin must be finite")
    return number


def _mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PatchControlArtifactError("INVALID_OBJECT", path, "value must be an object")
    return value


def _observation(value: object, path: str) -> PatchObservation:
    row = _mapping(value, path)
    return PatchObservation(
        site_id=_identifier(row.get("site_id"), f"{path}.site_id"),
        patched_margin=_finite(row.get("patched_margin"), f"{path}.patched_margin"),
    )


def _parse_cases(artifact: object, *, max_cases: int = 10_000) -> tuple[str, tuple[PatchControlCase, ...]]:
    root = _mapping(artifact, "$")
    experiment_id = _identifier(root.get("experiment_id"), "experiment_id")
    raw_cases = root.get("cases")
    if not isinstance(raw_cases, list):
        raise PatchControlArtifactError("INVALID_CASES", "cases", "cases must be an array")
    if len(raw_cases) > max_cases:
        raise PatchControlArtifactError(
            "CASE_BUDGET_EXCEEDED", "cases", f"artifact exceeds the {max_cases}-case budget"
        )
    cases: list[PatchControlCase] = []
    seen_case_ids: set[str] = set()
    for case_index, raw_case in enumerate(raw_cases):
        path = f"cases[{case_index}]"
        row = _mapping(raw_case, path)
        case_id = _identifier(row.get("case_id"), f"{path}.case_id")
        if case_id in seen_case_ids:
            raise PatchControlArtifactError("DUPLICATE_CASE_ID", f"{path}.case_id", "case IDs must be unique")
        seen_case_ids.add(case_id)
        target = _observation(row.get("target_patch"), f"{path}.target_patch")
        sham = _observation(row.get("sham_patch"), f"{path}.sham_patch")
        raw_controls = row.get("random_controls")
        if not isinstance(raw_controls, list):
            raise PatchControlArtifactError(
                "INVALID_RANDOM_CONTROLS",
                f"{path}.random_controls",
                "random_controls must be an array",
            )
        if len(raw_controls) > 1_000:
            raise PatchControlArtifactError(
                "CONTROL_BUDGET_EXCEEDED",
                f"{path}.random_controls",
                "a case cannot contain more than 1,000 random controls",
            )
        controls = tuple(
            _observation(value, f"{path}.random_controls[{index}]")
            for index, value in enumerate(raw_controls)
        )
        site_ids = [target.site_id, sham.site_id, *(item.site_id for item in controls)]
        if len(site_ids) != len(set(site_ids)):
            raise PatchControlArtifactError(
                "DUPLICATE_PATCH_SITE", path, "target, sham, and random patch sites must be unique"
            )
        cases.append(
            PatchControlCase(
                case_id=case_id,
                clean_margin=_finite(row.get("clean_margin"), f"{path}.clean_margin"),
                corrupted_margin=_finite(row.get("corrupted_margin"), f"{path}.corrupted_margin"),
                target_patch=target,
                sham_patch=sham,
                random_controls=controls,
            )
        )
    return experiment_id, tuple(cases)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _audit_case(case: PatchControlCase, policy: PatchControlPolicy) -> PatchControlCaseReport:
    signed_gap = case.clean_margin - case.corrupted_margin
    corruption_effect = abs(signed_gap)
    direction = 1.0 if signed_gap >= 0 else -1.0
    denominator = max(corruption_effect, policy.min_corruption_effect)
    target_recovery = direction * (case.target_patch.patched_margin - case.corrupted_margin) / denominator
    sham_effect = abs(case.sham_patch.patched_margin - case.corrupted_margin) / denominator
    control_effects = [
        abs(item.patched_margin - case.corrupted_margin) / denominator for item in case.random_controls
    ]
    control_p95 = _percentile(control_effects, 0.95) if control_effects else 0.0
    specificity_gap = target_recovery - control_p95

    reasons: list[str] = []
    if corruption_effect < policy.min_corruption_effect:
        reasons.append("INSUFFICIENT_CORRUPTION_EFFECT")
    if len(case.random_controls) < policy.min_random_controls:
        reasons.append("INSUFFICIENT_RANDOM_CONTROLS")
    if target_recovery < policy.min_target_recovery_fraction:
        reasons.append("TARGET_RECOVERY_BELOW_MINIMUM")
    if target_recovery > policy.max_target_recovery_fraction:
        reasons.append("TARGET_RECOVERY_ABOVE_MAXIMUM")
    if sham_effect > policy.max_sham_effect_fraction:
        reasons.append("SHAM_EFFECT_ABOVE_MAXIMUM")
    if control_p95 > policy.max_control_p95_effect_fraction:
        reasons.append("CONTROL_EFFECT_ABOVE_MAXIMUM")
    if specificity_gap < policy.min_specificity_gap:
        reasons.append("SPECIFICITY_GAP_BELOW_MINIMUM")
    return PatchControlCaseReport(
        case_id=case.case_id,
        target_site_id=case.target_patch.site_id,
        sham_site_id=case.sham_patch.site_id,
        corruption_effect=corruption_effect,
        target_recovery_fraction=target_recovery,
        sham_effect_fraction=sham_effect,
        control_p95_effect_fraction=control_p95,
        specificity_gap=specificity_gap,
        passed=not reasons,
        reason_codes=tuple(reasons),
    )


def _malformed(error: PatchControlArtifactError) -> PatchControlReport:
    return PatchControlReport(
        accepted=False,
        malformed=True,
        experiment_id=None,
        reason_codes=(error.code,),
        case_count=0,
        passing_cases=0,
        case_pass_rate=0.0,
        mean_target_recovery_fraction=None,
        median_target_recovery_fraction=None,
        mean_specificity_gap=None,
        cases=(),
        error_path=error.path,
        error_message=str(error),
    )


def audit_patch_controls(
    artifact: object,
    *,
    policy: PatchControlPolicy | None = None,
) -> PatchControlReport:
    """Audit target activation patches against sham and random-site controls."""
    selected_policy = policy or PatchControlPolicy()
    try:
        experiment_id, cases = _parse_cases(artifact)
    except PatchControlArtifactError as error:
        return _malformed(error)

    reports = tuple(_audit_case(case, selected_policy) for case in cases)
    passing = sum(item.passed for item in reports)
    pass_rate = passing / len(reports) if reports else 0.0
    reasons: list[str] = []
    if len(reports) < selected_policy.min_cases:
        reasons.append("INSUFFICIENT_CASES")
    if pass_rate < selected_policy.min_case_pass_rate:
        reasons.append("CASE_PASS_RATE_BELOW_MINIMUM")
    recoveries = [item.target_recovery_fraction for item in reports]
    specificity = [item.specificity_gap for item in reports]
    return PatchControlReport(
        accepted=not reasons,
        malformed=False,
        experiment_id=experiment_id,
        reason_codes=tuple(reasons),
        case_count=len(reports),
        passing_cases=passing,
        case_pass_rate=pass_rate,
        mean_target_recovery_fraction=mean(recoveries) if recoveries else None,
        median_target_recovery_fraction=median(recoveries) if recoveries else None,
        mean_specificity_gap=mean(specificity) if specificity else None,
        cases=reports,
    )


def build_case_from_scores(
    *,
    case_id: str,
    positive_candidate: str,
    negative_candidate: str,
    clean_scores: dict[str, float],
    corrupted_scores: dict[str, float],
    target_site_id: str,
    target_scores: dict[str, float],
    sham_site_id: str,
    sham_scores: dict[str, float],
    random_scores: dict[str, dict[str, float]],
) -> PatchControlCase:
    """Build a control case from the score dictionaries emitted by patch experiments."""

    def margin(scores: dict[str, float], path: str) -> float:
        try:
            positive = scores[positive_candidate]
            negative = scores[negative_candidate]
        except KeyError as exc:
            raise ValueError(f"{path} is missing candidate {exc.args[0]!r}") from exc
        return _finite(positive, f"{path}.{positive_candidate}") - _finite(
            negative, f"{path}.{negative_candidate}"
        )

    return PatchControlCase(
        case_id=_identifier(case_id, "case_id"),
        clean_margin=margin(clean_scores, "clean_scores"),
        corrupted_margin=margin(corrupted_scores, "corrupted_scores"),
        target_patch=PatchObservation(
            _identifier(target_site_id, "target_site_id"),
            margin(target_scores, "target_scores"),
        ),
        sham_patch=PatchObservation(
            _identifier(sham_site_id, "sham_site_id"), margin(sham_scores, "sham_scores")
        ),
        random_controls=tuple(
            PatchObservation(_identifier(site_id, "random_scores.site_id"), margin(scores, "random_scores"))
            for site_id, scores in sorted(random_scores.items())
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit activation-patching specificity against negative controls."
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--min-cases", type=int, default=3)
    parser.add_argument("--min-random-controls", type=int, default=5)
    parser.add_argument("--min-corruption-effect", type=float, default=0.5)
    parser.add_argument("--min-target-recovery", type=float, default=0.5)
    parser.add_argument("--max-target-recovery", type=float, default=1.5)
    parser.add_argument("--max-sham-effect", type=float, default=0.15)
    parser.add_argument("--max-control-p95-effect", type=float, default=0.3)
    parser.add_argument("--min-specificity-gap", type=float, default=0.25)
    parser.add_argument("--min-case-pass-rate", type=float, default=0.8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
        policy = PatchControlPolicy(
            min_cases=args.min_cases,
            min_random_controls=args.min_random_controls,
            min_corruption_effect=args.min_corruption_effect,
            min_target_recovery_fraction=args.min_target_recovery,
            max_target_recovery_fraction=args.max_target_recovery,
            max_sham_effect_fraction=args.max_sham_effect,
            max_control_p95_effect_fraction=args.max_control_p95_effect,
            min_specificity_gap=args.min_specificity_gap,
            min_case_pass_rate=args.min_case_pass_rate,
        )
        report = audit_patch_controls(artifact, policy=policy)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        report = _malformed(
            error
            if isinstance(error, PatchControlArtifactError)
            else PatchControlArtifactError("INVALID_INPUT", "$", str(error))
        )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if report.malformed:
        return 2
    return 0 if report.accepted else 3


if __name__ == "__main__":
    raise SystemExit(main())
