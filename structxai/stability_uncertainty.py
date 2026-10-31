from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist, mean

from structxai.stability import STABILITY_COHORT_SCHEMA

STABILITY_TASK_COHORT_SCHEMA = "struct-xai-stability-task-cohort/v1"
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_TASK_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_UINT32_RANGE = 1 << 32


class StabilityTaskCohortError(ValueError):
    """A task cohort is malformed or fails its prespecified uncertainty policy."""

    code = "STABILITY_TASK_COHORT_REJECTED"

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
        raise StabilityTaskCohortError("evidence is not canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def _bounded_number(name: str, value: object, *, lower: float, upper: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StabilityTaskCohortError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < lower or (upper is not None and number > upper):
        raise StabilityTaskCohortError(f"{name} is outside its finite bounds")
    return number


@dataclass(frozen=True, slots=True)
class StabilityTaskCohortPolicy:
    min_tasks: int = 8
    max_tasks: int = 128
    confidence_level: float = 0.95
    bootstrap_resamples: int = 2_000
    bootstrap_seed: int = 0
    min_accepted_task_rate_lower_bound: float = 0.60
    min_mean_task_pearson_lower_bound: float = 0.95
    min_mean_task_sign_agreement_lower_bound: float = 0.95
    max_mean_task_margin_drift_upper_bound: float = 0.05
    max_mean_task_final_margin_range_upper_bound: float = 0.05

    def __post_init__(self) -> None:
        if type(self.min_tasks) is not int or not 8 <= self.min_tasks <= 256:
            raise ValueError("min_tasks must be an integer from 8 through 256")
        if type(self.max_tasks) is not int or not self.min_tasks <= self.max_tasks <= 256:
            raise ValueError("max_tasks must be an integer from min_tasks through 256")
        if type(self.bootstrap_resamples) is not int or not 500 <= self.bootstrap_resamples <= 20_000:
            raise ValueError("bootstrap_resamples must be an integer from 500 through 20000")
        if type(self.bootstrap_seed) is not int or not 0 <= self.bootstrap_seed <= (1 << 64) - 1:
            raise ValueError("bootstrap_seed must be an unsigned 64-bit integer")
        for name, value, lower, upper in (
            ("confidence_level", self.confidence_level, 0.80, 0.999),
            (
                "min_accepted_task_rate_lower_bound",
                self.min_accepted_task_rate_lower_bound,
                0.0,
                1.0,
            ),
            (
                "min_mean_task_pearson_lower_bound",
                self.min_mean_task_pearson_lower_bound,
                -1.0,
                1.0,
            ),
            (
                "min_mean_task_sign_agreement_lower_bound",
                self.min_mean_task_sign_agreement_lower_bound,
                0.0,
                1.0,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or not lower <= float(value) <= upper:
                raise ValueError(f"{name} must be finite and between {lower} and {upper}")
        for name, value in (
            (
                "max_mean_task_margin_drift_upper_bound",
                self.max_mean_task_margin_drift_upper_bound,
            ),
            (
                "max_mean_task_final_margin_range_upper_bound",
                self.max_mean_task_final_margin_range_upper_bound,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_finding_codes(value: object, *, name: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str)
        or not item
        or len(item) > 128
        or not item.replace("_", "").isalnum()
        or item.upper() != item
        for item in value
    ):
        raise StabilityTaskCohortError(f"{name} must contain bounded canonical codes")
    if len(value) != len(set(value)):
        raise StabilityTaskCohortError(f"{name} must contain unique codes")
    return value


def _validated_task_report(report: object) -> dict[str, object]:
    if not isinstance(report, dict):
        raise StabilityTaskCohortError("each task report must be a dictionary")
    if report.get("schema_version") != STABILITY_COHORT_SCHEMA:
        raise StabilityTaskCohortError("task report has an unsupported stability schema")
    evidence_sha256 = report.get("evidence_sha256")
    if not isinstance(evidence_sha256, str) or _DIGEST_PATTERN.fullmatch(evidence_sha256) is None:
        raise StabilityTaskCohortError("task report evidence digest is not canonical")
    unsigned_report = {key: value for key, value in report.items() if key != "evidence_sha256"}
    if _canonical_digest(unsigned_report) != evidence_sha256:
        raise StabilityTaskCohortError("task report evidence digest does not match its content")

    for name in ("experiment_sha256", "experiment_fingerprint", "policy_sha256"):
        value = report.get(name)
        if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
            raise StabilityTaskCohortError(f"task report {name} is not a canonical SHA-256")
    policy = report.get("policy")
    if not isinstance(policy, dict) or _canonical_digest(policy) != report["policy_sha256"]:
        raise StabilityTaskCohortError("task report policy digest does not match its content")
    run_count = report.get("run_count")
    if type(run_count) is not int or not 3 <= run_count <= 64:
        raise StabilityTaskCohortError("task report run_count is outside the supported bounds")
    accepted = report.get("accepted")
    if type(accepted) is not bool:
        raise StabilityTaskCohortError("task report accepted verdict must be boolean")
    finding_codes = _validate_finding_codes(report.get("finding_codes"), name="task finding_codes")

    pairs = report.get("pairs")
    if not isinstance(pairs, list) or not 1 <= len(pairs) <= 120:
        raise StabilityTaskCohortError("task report must contain a bounded non-empty pair list")
    seen_pairs: set[tuple[str, str]] = set()
    pearsons: list[float] = []
    sign_agreements: list[float] = []
    margin_drifts: list[float] = []
    final_ranges: list[float] = []
    pair_findings: set[str] = set()
    for pair in pairs:
        if not isinstance(pair, dict):
            raise StabilityTaskCohortError("task report pair entries must be dictionaries")
        positive = pair.get("positive")
        negative = pair.get("negative")
        if any(
            not isinstance(label, str)
            or not label
            or len(label.encode("utf-8")) > 256
            or any(ord(character) < 32 for character in label)
            for label in (positive, negative)
        ):
            raise StabilityTaskCohortError("task report pair labels must be bounded strings")
        pair_id = (positive, negative)
        if positive == negative or pair_id in seen_pairs:
            raise StabilityTaskCohortError("task report pair identities must be unique")
        seen_pairs.add(pair_id)
        pearsons.append(
            _bounded_number(
                "minimum_pairwise_pearson",
                pair.get("minimum_pairwise_pearson"),
                lower=-1.0,
                upper=1.0,
            )
        )
        sign_agreements.append(
            _bounded_number(
                "minimum_pairwise_sign_agreement",
                pair.get("minimum_pairwise_sign_agreement"),
                lower=0.0,
                upper=1.0,
            )
        )
        margin_drifts.append(
            _bounded_number(
                "maximum_pairwise_mean_abs_margin_difference",
                pair.get("maximum_pairwise_mean_abs_margin_difference"),
                lower=0.0,
            )
        )
        final_ranges.append(
            _bounded_number(
                "final_margin_range",
                pair.get("final_margin_range"),
                lower=0.0,
            )
        )
        qualifies = pair.get("qualifies")
        if type(qualifies) is not bool:
            raise StabilityTaskCohortError("task report pair qualifies verdict must be boolean")
        findings = _validate_finding_codes(pair.get("findings"), name="pair findings")
        if qualifies != (not findings):
            raise StabilityTaskCohortError("task report pair verdict conflicts with its findings")
        pair_findings.update(findings)
    if finding_codes != sorted(pair_findings) or accepted != (not finding_codes):
        raise StabilityTaskCohortError("task report verdict conflicts with its pair findings")

    return {
        "evidence_sha256": evidence_sha256,
        "experiment_sha256": report["experiment_sha256"],
        "experiment_fingerprint": report["experiment_fingerprint"],
        "policy_sha256": report["policy_sha256"],
        "run_count": run_count,
        "accepted": accepted,
        "minimum_pairwise_pearson": min(pearsons),
        "minimum_pairwise_sign_agreement": min(sign_agreements),
        "maximum_pairwise_margin_drift": max(margin_drifts),
        "maximum_final_margin_range": max(final_ranges),
    }


class _DeterministicIndexStream:
    def __init__(self, seed: int, size: int) -> None:
        self._seed = seed.to_bytes(8, "big")
        self._size = size
        self._counter = 0
        self._limit = (_UINT32_RANGE // size) * size

    def next(self) -> int:
        while True:
            digest = hashlib.sha256(
                b"struct-xai/task-bootstrap/v1\x00" + self._seed + self._counter.to_bytes(16, "big")
            ).digest()
            self._counter += 1
            value = int.from_bytes(digest[:4], "big")
            if value < self._limit:
                return value % self._size


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = position - lower_index
    return sorted_values[lower_index] * (1.0 - weight) + sorted_values[upper_index] * weight


def _bootstrap_mean_intervals(
    columns: Mapping[str, Sequence[float]],
    *,
    confidence_level: float,
    resamples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    names = tuple(columns)
    size = len(next(iter(columns.values())))
    if any(len(values) != size for values in columns.values()):
        raise AssertionError("bootstrap columns must have equal lengths")
    stream = _DeterministicIndexStream(seed, size)
    distributions = {name: [] for name in names}
    for _ in range(resamples):
        indices = [stream.next() for _ in range(size)]
        for name in names:
            values = columns[name]
            distributions[name].append(mean(values[index] for index in indices))
    alpha = (1.0 - confidence_level) / 2.0
    return {
        name: {
            "estimate": mean(columns[name]),
            "lower": _percentile(sorted(distributions[name]), alpha),
            "upper": _percentile(sorted(distributions[name]), 1.0 - alpha),
        }
        for name in names
    }


def _wilson_interval(successes: int, total: int, confidence_level: float) -> dict[str, float]:
    estimate = successes / total
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(estimate * (1.0 - estimate) / total + z * z / (4.0 * total * total)) / denominator
    return {"estimate": estimate, "lower": center - radius, "upper": center + radius}


def audit_stability_task_cohort(
    task_reports: Mapping[str, dict],
    *,
    policy: StabilityTaskCohortPolicy | None = None,
) -> dict[str, object]:
    """Quantify task-to-task uncertainty without treating dependent run pairs as samples."""

    active_policy = policy or StabilityTaskCohortPolicy()
    if not isinstance(active_policy, StabilityTaskCohortPolicy):
        raise TypeError("policy must be a StabilityTaskCohortPolicy")
    if not isinstance(task_reports, Mapping):
        raise TypeError("task_reports must be a mapping")
    if not active_policy.min_tasks <= len(task_reports) <= active_policy.max_tasks:
        raise StabilityTaskCohortError("task count is outside the policy bounds")
    task_ids = tuple(task_reports)
    if any(
        not isinstance(task_id, str) or _TASK_ID_PATTERN.fullmatch(task_id) is None for task_id in task_ids
    ):
        raise StabilityTaskCohortError("task IDs must be canonical identifiers")
    if len(task_reports) * active_policy.bootstrap_resamples > 5_000_000:
        raise StabilityTaskCohortError("task bootstrap exceeds the work budget")

    validated = {task_id: _validated_task_report(task_reports[task_id]) for task_id in sorted(task_ids)}
    policy_hashes = {str(item["policy_sha256"]) for item in validated.values()}
    run_counts = {int(item["run_count"]) for item in validated.values()}
    experiment_hashes = [str(item["experiment_sha256"]) for item in validated.values()]
    fingerprints = [str(item["experiment_fingerprint"]) for item in validated.values()]
    if len(policy_hashes) != 1:
        raise StabilityTaskCohortError("task reports use different run-cohort policies")
    if len(run_counts) != 1:
        raise StabilityTaskCohortError("task reports use different run counts")
    if len(set(experiment_hashes)) != len(experiment_hashes) or len(set(fingerprints)) != len(fingerprints):
        raise StabilityTaskCohortError("task cohort contains duplicate experiment commitments")

    ordered_items = tuple(validated.items())
    columns = {
        "mean_task_minimum_pairwise_pearson": tuple(
            float(item["minimum_pairwise_pearson"]) for _, item in ordered_items
        ),
        "mean_task_minimum_pairwise_sign_agreement": tuple(
            float(item["minimum_pairwise_sign_agreement"]) for _, item in ordered_items
        ),
        "mean_task_maximum_pairwise_margin_drift": tuple(
            float(item["maximum_pairwise_margin_drift"]) for _, item in ordered_items
        ),
        "mean_task_maximum_final_margin_range": tuple(
            float(item["maximum_final_margin_range"]) for _, item in ordered_items
        ),
    }
    intervals = _bootstrap_mean_intervals(
        columns,
        confidence_level=float(active_policy.confidence_level),
        resamples=active_policy.bootstrap_resamples,
        seed=active_policy.bootstrap_seed,
    )
    intervals["accepted_task_rate"] = _wilson_interval(
        sum(bool(item["accepted"]) for _, item in ordered_items),
        len(ordered_items),
        float(active_policy.confidence_level),
    )

    findings: list[str] = []
    if intervals["accepted_task_rate"]["lower"] < active_policy.min_accepted_task_rate_lower_bound:
        findings.append("ACCEPTED_TASK_RATE_LOWER_BOUND_BELOW_FLOOR")
    if (
        intervals["mean_task_minimum_pairwise_pearson"]["lower"]
        < active_policy.min_mean_task_pearson_lower_bound
    ):
        findings.append("MEAN_TASK_PEARSON_LOWER_BOUND_BELOW_FLOOR")
    if (
        intervals["mean_task_minimum_pairwise_sign_agreement"]["lower"]
        < active_policy.min_mean_task_sign_agreement_lower_bound
    ):
        findings.append("MEAN_TASK_SIGN_AGREEMENT_LOWER_BOUND_BELOW_FLOOR")
    if (
        intervals["mean_task_maximum_pairwise_margin_drift"]["upper"]
        > active_policy.max_mean_task_margin_drift_upper_bound
    ):
        findings.append("MEAN_TASK_MARGIN_DRIFT_UPPER_BOUND_ABOVE_LIMIT")
    if (
        intervals["mean_task_maximum_final_margin_range"]["upper"]
        > active_policy.max_mean_task_final_margin_range_upper_bound
    ):
        findings.append("MEAN_TASK_FINAL_MARGIN_RANGE_UPPER_BOUND_ABOVE_LIMIT")

    policy_document = active_policy.to_audit_dict()
    report: dict[str, object] = {
        "schema_version": STABILITY_TASK_COHORT_SCHEMA,
        "method": "task-resampled-percentile-bootstrap-with-wilson-rate",
        "resampling_unit": "task",
        "estimand": "mean task-level worst-pair stability metric",
        "policy": policy_document,
        "policy_sha256": _canonical_digest(policy_document),
        "run_cohort_policy_sha256": next(iter(policy_hashes)),
        "task_count": len(ordered_items),
        "run_count_per_task": next(iter(run_counts)),
        "bootstrap_work_units": len(ordered_items) * active_policy.bootstrap_resamples,
        "tasks": [
            {
                "task_id": task_id,
                "experiment_sha256": item["experiment_sha256"],
                "experiment_fingerprint": item["experiment_fingerprint"],
                "stability_evidence_sha256": item["evidence_sha256"],
            }
            for task_id, item in ordered_items
        ],
        "intervals": intervals,
        "accepted": not findings,
        "finding_codes": findings,
    }
    report["evidence_sha256"] = _canonical_digest(report)
    return report


def require_stability_task_cohort(
    task_reports: Mapping[str, dict],
    *,
    policy: StabilityTaskCohortPolicy | None = None,
) -> dict[str, object]:
    """Return accepted task-level uncertainty evidence or raise one stable error."""

    report = audit_stability_task_cohort(task_reports, policy=policy)
    if not report["accepted"]:
        findings = report["finding_codes"]
        assert isinstance(findings, list)
        raise StabilityTaskCohortError("policy failed: " + ",".join(str(item) for item in findings))
    return report
