from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from statistics import fmean

from structxai.patch_control_audit import PatchControlPolicy, audit_patch_control_specificity
from structxai.patch_control_cohort import (
    PatchControlCohortError,
    PatchControlCohortPolicy,
    PatchControlTask,
    audit_patch_control_cohort,
)
from structxai.patch_dose_response import DoseResponseAuditPolicy

GROUP_BOOTSTRAP_SCHEMA_VERSION = 1
GROUP_BOOTSTRAP_METHOD = "template-group-percentile-bootstrap-bonferroni/v1"
_GROUP_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_UINT32_RANGE = 1 << 32


class PatchControlGroupBootstrapError(PatchControlCohortError):
    """Grouped control evidence is malformed or exceeds a bounded work policy."""


class PatchControlGroupBootstrapRejected(PatchControlGroupBootstrapError):
    """Valid grouped evidence failed its prespecified promotion policy."""


@dataclass(frozen=True, slots=True)
class PatchControlGroupBootstrapPolicy:
    min_groups: int = 4
    max_groups: int = 128
    max_tasks_per_group: int = 64
    family_confidence: float = 0.95
    bootstrap_resamples: int = 2_000
    bootstrap_seed: int = 0
    min_group_specific_fraction_lower_bound: float = 0.50
    min_group_specific_layers: int = 1

    def __post_init__(self) -> None:
        for name, value, lower, upper in (
            ("min_groups", self.min_groups, 4, 256),
            ("max_groups", self.max_groups, 4, 256),
            ("max_tasks_per_group", self.max_tasks_per_group, 1, 512),
            ("bootstrap_resamples", self.bootstrap_resamples, 500, 20_000),
            ("min_group_specific_layers", self.min_group_specific_layers, 1, 1_024),
        ):
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer between {lower} and {upper}")
        if self.min_groups > self.max_groups:
            raise ValueError("min_groups cannot exceed max_groups")
        if type(self.bootstrap_seed) is not int or not 0 <= self.bootstrap_seed <= (1 << 64) - 1:
            raise ValueError("bootstrap_seed must be an unsigned 64-bit integer")
        for name, value, lower, upper in (
            ("family_confidence", self.family_confidence, 0.80, 0.999),
            (
                "min_group_specific_fraction_lower_bound",
                self.min_group_specific_fraction_lower_bound,
                0.0,
                1.0,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not lower <= float(value) <= upper
            ):
                raise ValueError(f"{name} must be finite and between {lower} and {upper}")

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GroupedTaskEvidence:
    task_id_sha256: str
    group_id_sha256: str
    control_audit_sha256: str


@dataclass(frozen=True, slots=True)
class GroupEvidence:
    group_id_sha256: str
    task_count: int
    task_inventory_sha256: str


@dataclass(frozen=True, slots=True)
class GroupBootstrapLayerAssessment:
    layer: int
    specific_tasks: int
    total_tasks: int
    observed_task_fraction: float
    observed_group_mean_fraction: float
    bootstrap_lower_bound: float
    group_specific: bool
    finding_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PatchControlGroupBootstrapAudit:
    schema_version: int
    status: str
    method: str
    resampling_unit: str
    base_cohort_evidence_sha256: str
    group_policy_sha256: str
    family_confidence: float
    hypothesis_count: int
    per_hypothesis_alpha: float
    bootstrap_resamples: int
    bootstrap_work_units: int
    tasks: tuple[GroupedTaskEvidence, ...]
    groups: tuple[GroupEvidence, ...]
    total_layers: int
    group_specific_layers: int
    layers: tuple[GroupBootstrapLayerAssessment, ...]
    finding_codes: tuple[str, ...]
    evidence_sha256: str

    def body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "method": self.method,
            "resampling_unit": self.resampling_unit,
            "base_cohort_evidence_sha256": self.base_cohort_evidence_sha256,
            "group_policy_sha256": self.group_policy_sha256,
            "family_confidence": self.family_confidence,
            "hypothesis_count": self.hypothesis_count,
            "per_hypothesis_alpha": self.per_hypothesis_alpha,
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_work_units": self.bootstrap_work_units,
            "tasks": [asdict(task) for task in self.tasks],
            "groups": [asdict(group) for group in self.groups],
            "total_layers": self.total_layers,
            "group_specific_layers": self.group_specific_layers,
            "layers": [asdict(layer) for layer in self.layers],
            "finding_codes": list(self.finding_codes),
        }

    def as_dict(self) -> dict[str, object]:
        return {**self.body(), "evidence_sha256": self.evidence_sha256}

    def with_digest(self) -> PatchControlGroupBootstrapAudit:
        return replace(self, evidence_sha256=_digest(self.body()))


def _digest(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise PatchControlGroupBootstrapError("NON_CANONICAL_GROUP_BOOTSTRAP_EVIDENCE") from exc
    return hashlib.sha256(payload).hexdigest()


def _identifier_digest(domain: bytes, identifier: str) -> str:
    return hashlib.sha256(domain + b"\x00" + identifier.encode("ascii")).hexdigest()


class _DeterministicGroupIndexStream:
    def __init__(self, seed: int, size: int) -> None:
        self._seed = seed.to_bytes(8, "big")
        self._size = size
        self._counter = 0
        self._limit = (_UINT32_RANGE // size) * size

    def next(self) -> int:
        while True:
            digest = hashlib.sha256(
                b"struct-xai/patch-control-group-bootstrap/v1\x00"
                + self._seed
                + self._counter.to_bytes(16, "big")
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


def audit_patch_control_group_bootstrap(
    tasks: Mapping[str, PatchControlTask],
    task_groups: Mapping[str, str],
    *,
    dose_policy: DoseResponseAuditPolicy | None = None,
    control_policy: PatchControlPolicy | None = None,
    cohort_policy: PatchControlCohortPolicy | None = None,
    group_policy: PatchControlGroupBootstrapPolicy | None = None,
) -> PatchControlGroupBootstrapAudit:
    """Resample prespecified template groups instead of treating related tasks as independent."""

    active_group_policy = group_policy or PatchControlGroupBootstrapPolicy()
    if not isinstance(active_group_policy, PatchControlGroupBootstrapPolicy):
        raise TypeError("group_policy must be a PatchControlGroupBootstrapPolicy")
    if not isinstance(task_groups, Mapping):
        raise PatchControlGroupBootstrapError("INVALID_TASK_GROUPS")

    base_audit = audit_patch_control_cohort(
        tasks,
        dose_policy=dose_policy,
        control_policy=control_policy,
        cohort_policy=cohort_policy,
    )
    if set(task_groups) != set(tasks):
        raise PatchControlGroupBootstrapError("TASK_GROUP_INVENTORY_MISMATCH")
    if any(
        not isinstance(group_id, str) or _GROUP_ID_RE.fullmatch(group_id) is None
        for group_id in task_groups.values()
    ):
        raise PatchControlGroupBootstrapError("INVALID_TASK_GROUP_ID")

    group_members: dict[str, list[str]] = defaultdict(list)
    for task_id in sorted(tasks):
        group_members[task_groups[task_id]].append(task_id)
    if not active_group_policy.min_groups <= len(group_members) <= active_group_policy.max_groups:
        raise PatchControlGroupBootstrapError("TASK_GROUP_COUNT_OUT_OF_BOUNDS")
    if any(len(members) > active_group_policy.max_tasks_per_group for members in group_members.values()):
        raise PatchControlGroupBootstrapError("TASKS_PER_GROUP_OUT_OF_BOUNDS")

    ordered_tasks = tuple(sorted(tasks))
    task_audits = {
        task_id: audit_patch_control_specificity(
            tasks[task_id].treatment,
            tasks[task_id].controls,
            dose_policy=dose_policy,
            control_policy=control_policy,
        )
        for task_id in ordered_tasks
    }
    layer_ids = tuple(layer.layer for layer in next(iter(task_audits.values())).layers)
    group_ids = tuple(sorted(group_members))
    work_units = len(group_ids) * active_group_policy.bootstrap_resamples * len(layer_ids)
    if work_units > 5_000_000:
        raise PatchControlGroupBootstrapError("GROUP_BOOTSTRAP_WORK_BUDGET_EXCEEDED")

    task_evidence = tuple(
        GroupedTaskEvidence(
            task_id_sha256=_identifier_digest(b"struct-xai/patch-control-grouped-task/v1", task_id),
            group_id_sha256=_identifier_digest(
                b"struct-xai/patch-control-template-group/v1", task_groups[task_id]
            ),
            control_audit_sha256=task_audits[task_id].evidence_sha256,
        )
        for task_id in ordered_tasks
    )
    task_hash_by_id = {task_id: item.task_id_sha256 for task_id, item in zip(ordered_tasks, task_evidence)}
    group_evidence = tuple(
        GroupEvidence(
            group_id_sha256=_identifier_digest(b"struct-xai/patch-control-template-group/v1", group_id),
            task_count=len(group_members[group_id]),
            task_inventory_sha256=_digest(
                {"task_id_sha256": sorted(task_hash_by_id[item] for item in group_members[group_id])}
            ),
        )
        for group_id in group_ids
    )

    group_means_by_layer: list[tuple[float, ...]] = []
    for index, layer_id in enumerate(layer_ids):
        means = []
        for group_id in group_ids:
            values = []
            for task_id in group_members[group_id]:
                layer = task_audits[task_id].layers[index]
                if layer.layer != layer_id:
                    raise PatchControlGroupBootstrapError("GROUPED_COHORT_LAYER_MISMATCH")
                values.append(float(layer.specific))
            means.append(fmean(values))
        group_means_by_layer.append(tuple(means))

    stream = _DeterministicGroupIndexStream(active_group_policy.bootstrap_seed, len(group_ids))
    distributions: list[list[float]] = [[] for _ in layer_ids]
    for _ in range(active_group_policy.bootstrap_resamples):
        indices = tuple(stream.next() for _ in group_ids)
        for layer_index, group_means in enumerate(group_means_by_layer):
            distributions[layer_index].append(fmean(group_means[index] for index in indices))

    per_hypothesis_alpha = (1.0 - float(active_group_policy.family_confidence)) / len(layer_ids)
    layers: list[GroupBootstrapLayerAssessment] = []
    for index, layer_id in enumerate(layer_ids):
        specific_tasks = sum(task_audits[task_id].layers[index].specific for task_id in ordered_tasks)
        lower_bound = _percentile(sorted(distributions[index]), per_hypothesis_alpha)
        findings = (
            ()
            if lower_bound >= active_group_policy.min_group_specific_fraction_lower_bound
            else ("GROUP_SPECIFIC_FRACTION_LOWER_BOUND_BELOW_MINIMUM",)
        )
        layers.append(
            GroupBootstrapLayerAssessment(
                layer=layer_id,
                specific_tasks=specific_tasks,
                total_tasks=len(ordered_tasks),
                observed_task_fraction=specific_tasks / len(ordered_tasks),
                observed_group_mean_fraction=fmean(group_means_by_layer[index]),
                bootstrap_lower_bound=lower_bound,
                group_specific=not findings,
                finding_codes=findings,
            )
        )

    group_specific_layers = sum(layer.group_specific for layer in layers)
    global_findings: list[str] = []
    if base_audit.status != "accepted":
        global_findings.append("BASE_TASK_COHORT_REJECTED")
    if group_specific_layers < active_group_policy.min_group_specific_layers:
        global_findings.append("GROUP_SPECIFIC_LAYER_COUNT_BELOW_MINIMUM")
    result = PatchControlGroupBootstrapAudit(
        schema_version=GROUP_BOOTSTRAP_SCHEMA_VERSION,
        status="accepted" if not global_findings else "rejected",
        method=GROUP_BOOTSTRAP_METHOD,
        resampling_unit="template_group",
        base_cohort_evidence_sha256=base_audit.evidence_sha256,
        group_policy_sha256=_digest(active_group_policy.as_dict()),
        family_confidence=float(active_group_policy.family_confidence),
        hypothesis_count=len(layer_ids),
        per_hypothesis_alpha=per_hypothesis_alpha,
        bootstrap_resamples=active_group_policy.bootstrap_resamples,
        bootstrap_work_units=work_units,
        tasks=task_evidence,
        groups=group_evidence,
        total_layers=len(layer_ids),
        group_specific_layers=group_specific_layers,
        layers=tuple(layers),
        finding_codes=tuple(global_findings),
        evidence_sha256="",
    )
    return result.with_digest()


def require_patch_control_group_bootstrap(
    tasks: Mapping[str, PatchControlTask],
    task_groups: Mapping[str, str],
    *,
    dose_policy: DoseResponseAuditPolicy | None = None,
    control_policy: PatchControlPolicy | None = None,
    cohort_policy: PatchControlCohortPolicy | None = None,
    group_policy: PatchControlGroupBootstrapPolicy | None = None,
) -> PatchControlGroupBootstrapAudit:
    """Return accepted grouped evidence or raise one stable promotion error."""

    audit = audit_patch_control_group_bootstrap(
        tasks,
        task_groups,
        dose_policy=dose_policy,
        control_policy=control_policy,
        cohort_policy=cohort_policy,
        group_policy=group_policy,
    )
    if audit.status != "accepted":
        raise PatchControlGroupBootstrapRejected("PATCH_CONTROL_GROUP_BOOTSTRAP_REJECTED")
    return audit
