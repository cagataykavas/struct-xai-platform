from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class AttributionCase:
    case_id: str
    feature_ids: tuple[str, ...]
    scores: tuple[float, ...]


@dataclass(frozen=True)
class RandomizationStage:
    stage_id: str
    randomized_fraction: float
    cases: tuple[AttributionCase, ...]


@dataclass(frozen=True)
class RandomizationPolicy:
    top_k: int = 5
    min_cases: int = 20
    min_stages: int = 2
    min_final_randomized_fraction: float = 0.9
    max_final_abs_cosine: float = 0.3
    max_final_top_k_overlap: float = 0.4
    max_final_sign_agreement: float = 0.65
    max_stage_cosine_recovery: float = 0.05

    def __post_init__(self) -> None:
        if self.top_k < 1 or self.min_cases < 1 or self.min_stages < 1:
            raise ValueError("top_k and evidence minimums must be positive")
        for name in (
            "min_final_randomized_fraction",
            "max_final_abs_cosine",
            "max_final_top_k_overlap",
            "max_final_sign_agreement",
            "max_stage_cosine_recovery",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and between 0 and 1")


@dataclass(frozen=True)
class StageSummary:
    stage_id: str
    randomized_fraction: float
    mean_abs_cosine: float
    mean_top_k_overlap: float
    mean_top_k_sign_agreement: float


@dataclass(frozen=True)
class FinalCaseEvidence:
    case_id: str
    abs_cosine: float
    top_k_overlap: float
    top_k_sign_agreement: float


@dataclass(frozen=True)
class RandomizationReport:
    passed: bool
    reasons: tuple[str, ...]
    case_count: int
    stage_count: int
    top_k: int
    stages: tuple[StageSummary, ...]
    final_case_evidence: tuple[FinalCaseEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "case_count": self.case_count,
            "stage_count": self.stage_count,
            "top_k": self.top_k,
            "stages": [asdict(stage) for stage in self.stages],
            "final_case_evidence": [asdict(row) for row in self.final_case_evidence],
        }


@dataclass(frozen=True)
class _CaseMetrics:
    abs_cosine: float
    top_k_overlap: float
    top_k_sign_agreement: float


def _validate_cases(cases: tuple[AttributionCase, ...], *, label: str) -> dict[str, AttributionCase]:
    indexed: dict[str, AttributionCase] = {}
    for case in cases:
        if not isinstance(case, AttributionCase):
            raise TypeError(f"{label} cases must be AttributionCase instances")
        if not isinstance(case.case_id, str) or not case.case_id or case.case_id != case.case_id.strip():
            raise ValueError(f"{label} case_id must be non-empty and trimmed")
        if case.case_id in indexed:
            raise ValueError(f"duplicate {label} case_id: {case.case_id}")
        if not case.feature_ids or len(case.feature_ids) != len(case.scores):
            raise ValueError(f"{label} case {case.case_id} has misaligned features and scores")
        if len(set(case.feature_ids)) != len(case.feature_ids) or any(
            not isinstance(feature_id, str) or not feature_id for feature_id in case.feature_ids
        ):
            raise ValueError(f"{label} case {case.case_id} has invalid feature ids")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in case.scores
        ):
            raise ValueError(f"{label} case {case.case_id} has non-finite attribution scores")
        if not any(value != 0 for value in case.scores):
            raise ValueError(f"{label} case {case.case_id} has a zero attribution vector")
        indexed[case.case_id] = case
    return indexed


def _top_indices(case: AttributionCase, k: int) -> tuple[int, ...]:
    ordered = sorted(
        range(len(case.scores)),
        key=lambda index: (-abs(case.scores[index]), case.feature_ids[index]),
    )
    return tuple(ordered[: min(k, len(ordered))])


def _sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def _metrics(
    baseline: AttributionCase,
    randomized: AttributionCase,
    *,
    top_k: int,
) -> _CaseMetrics:
    left_norm = math.sqrt(sum(value * value for value in baseline.scores))
    right_norm = math.sqrt(sum(value * value for value in randomized.scores))
    cosine = sum(left * right for left, right in zip(baseline.scores, randomized.scores, strict=True)) / (
        left_norm * right_norm
    )
    baseline_top = _top_indices(baseline, top_k)
    randomized_top = set(_top_indices(randomized, top_k))
    overlap = len(set(baseline_top) & randomized_top) / len(baseline_top)
    sign_agreement = mean(
        _sign(baseline.scores[index]) == _sign(randomized.scores[index]) for index in baseline_top
    )
    return _CaseMetrics(abs(cosine), overlap, sign_agreement)


def evaluate_randomization_sanity(
    baseline: tuple[AttributionCase, ...],
    stages: tuple[RandomizationStage, ...],
    policy: RandomizationPolicy | None = None,
) -> RandomizationReport:
    policy = policy or RandomizationPolicy()
    baseline_by_id = _validate_cases(baseline, label="baseline")
    if not stages:
        raise ValueError("at least one randomization stage is required")

    stage_ids: set[str] = set()
    previous_fraction = 0.0
    summaries: list[StageSummary] = []
    per_stage_metrics: list[dict[str, _CaseMetrics]] = []
    for stage in stages:
        if not isinstance(stage, RandomizationStage):
            raise TypeError("stages must be RandomizationStage instances")
        if not stage.stage_id or stage.stage_id != stage.stage_id.strip():
            raise ValueError("stage_id must be non-empty and trimmed")
        if stage.stage_id in stage_ids:
            raise ValueError(f"duplicate stage_id: {stage.stage_id}")
        if (
            not math.isfinite(stage.randomized_fraction)
            or not previous_fraction < stage.randomized_fraction <= 1
        ):
            raise ValueError("randomized fractions must be finite, positive and strictly increasing")
        stage_ids.add(stage.stage_id)
        previous_fraction = stage.randomized_fraction

        randomized_by_id = _validate_cases(stage.cases, label=stage.stage_id)
        if set(randomized_by_id) != set(baseline_by_id):
            raise ValueError(f"stage {stage.stage_id} must contain the baseline case ids")
        stage_metrics: dict[str, _CaseMetrics] = {}
        for case_id, baseline_case in baseline_by_id.items():
            randomized_case = randomized_by_id[case_id]
            if randomized_case.feature_ids != baseline_case.feature_ids:
                raise ValueError(f"feature alignment differs for case {case_id}")
            stage_metrics[case_id] = _metrics(
                baseline_case,
                randomized_case,
                top_k=policy.top_k,
            )
        per_stage_metrics.append(stage_metrics)
        summaries.append(
            StageSummary(
                stage_id=stage.stage_id,
                randomized_fraction=stage.randomized_fraction,
                mean_abs_cosine=mean(row.abs_cosine for row in stage_metrics.values()),
                mean_top_k_overlap=mean(row.top_k_overlap for row in stage_metrics.values()),
                mean_top_k_sign_agreement=mean(row.top_k_sign_agreement for row in stage_metrics.values()),
            )
        )

    reasons: list[str] = []
    if len(baseline_by_id) < policy.min_cases:
        reasons.append("insufficient_cases")
    if len(stages) < policy.min_stages:
        reasons.append("insufficient_randomization_stages")
    final = summaries[-1]
    if final.randomized_fraction < policy.min_final_randomized_fraction:
        reasons.append("insufficient_final_randomization")
    if final.mean_abs_cosine > policy.max_final_abs_cosine:
        reasons.append("final_cosine_too_high")
    if final.mean_top_k_overlap > policy.max_final_top_k_overlap:
        reasons.append("final_top_k_overlap_too_high")
    if final.mean_top_k_sign_agreement > policy.max_final_sign_agreement:
        reasons.append("final_sign_agreement_too_high")
    if any(
        current.mean_abs_cosine > previous.mean_abs_cosine + policy.max_stage_cosine_recovery
        for previous, current in pairwise(summaries)
    ):
        reasons.append("cosine_recovery_exceeded")

    final_metrics = per_stage_metrics[-1]
    evidence = tuple(
        FinalCaseEvidence(case_id, **asdict(final_metrics[case_id])) for case_id in sorted(final_metrics)
    )
    return RandomizationReport(
        passed=not reasons,
        reasons=tuple(reasons),
        case_count=len(baseline_by_id),
        stage_count=len(stages),
        top_k=policy.top_k,
        stages=tuple(summaries),
        final_case_evidence=evidence,
    )


def _case(payload: dict[str, Any]) -> AttributionCase:
    return AttributionCase(
        case_id=payload["case_id"],
        feature_ids=tuple(payload["feature_ids"]),
        scores=tuple(payload["scores"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit attribution model sensitivity")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.artifact.read_text(encoding="utf-8"))
        report = evaluate_randomization_sanity(
            tuple(_case(row) for row in payload["baseline"]),
            tuple(
                RandomizationStage(
                    stage_id=stage["stage_id"],
                    randomized_fraction=stage["randomized_fraction"],
                    cases=tuple(_case(row) for row in stage["cases"]),
                )
                for stage in payload["stages"]
            ),
            RandomizationPolicy(**payload.get("policy", {})),
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(report.to_dict(), sort_keys=True))
    return 2 if args.require_pass and not report.passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
