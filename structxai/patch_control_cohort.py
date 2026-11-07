from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from statistics import NormalDist, fmean

from structxai.patch_control_audit import (
    PatchControlAuditError,
    PatchControlPolicy,
    audit_patch_control_specificity,
)
from structxai.patch_dose_response import DoseResponseAuditPolicy, PatchSweepResult

COHORT_SCHEMA_VERSION = 1
BOUND_METHOD = "wilson-bonferroni-one-sided/v1"
_TASK_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


class PatchControlCohortError(PatchControlAuditError):
    """A control cohort is malformed, incompatible, or statistically unsafe."""


class PatchControlCohortRejected(PatchControlCohortError):
    """Valid cohort evidence failed its prespecified promotion policy."""


@dataclass(frozen=True, slots=True)
class PatchControlTask:
    treatment: PatchSweepResult
    controls: Mapping[str, PatchSweepResult]


@dataclass(frozen=True, slots=True)
class PatchControlCohortPolicy:
    min_tasks: int = 8
    max_tasks: int = 512
    family_confidence: float = 0.95
    min_specific_task_fraction_lower_bound: float = 0.50
    min_cohort_specific_layers: int = 1

    def __post_init__(self) -> None:
        for name, value, lower, upper in (
            ("min_tasks", self.min_tasks, 4, 512),
            ("max_tasks", self.max_tasks, 4, 512),
            ("min_cohort_specific_layers", self.min_cohort_specific_layers, 1, 1_024),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"{name} must be between {lower} and {upper}")
        if self.min_tasks > self.max_tasks:
            raise ValueError("min_tasks cannot exceed max_tasks")
        for name, value, lower, upper in (
            ("family_confidence", self.family_confidence, 0.80, 0.999),
            (
                "min_specific_task_fraction_lower_bound",
                self.min_specific_task_fraction_lower_bound,
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
class CohortTaskEvidence:
    task_id_sha256: str
    target_input_sha256: str
    treatment_source_input_sha256: str
    treatment_sweep_evidence_sha256: str
    control_audit_sha256: str
    control_count: int


@dataclass(frozen=True, slots=True)
class CohortLayerAssessment:
    layer: int
    specific_tasks: int
    total_tasks: int
    observed_specific_fraction: float
    simultaneous_lower_bound: float
    mean_endpoint_excess: float
    mean_auc_excess: float
    cohort_specific: bool
    finding_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PatchControlCohortAudit:
    schema_version: int
    status: str
    bound_method: str
    model_revision: str
    sweep_spec_sha256: str
    sweep_policy_sha256: str
    dose_policy_sha256: str
    control_policy_sha256: str
    cohort_policy_sha256: str
    family_confidence: float
    hypothesis_count: int
    per_hypothesis_alpha: float
    tasks: tuple[CohortTaskEvidence, ...]
    total_layers: int
    cohort_specific_layers: int
    layers: tuple[CohortLayerAssessment, ...]
    finding_codes: tuple[str, ...]
    evidence_sha256: str

    def body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "bound_method": self.bound_method,
            "model_revision": self.model_revision,
            "sweep_spec_sha256": self.sweep_spec_sha256,
            "sweep_policy_sha256": self.sweep_policy_sha256,
            "dose_policy_sha256": self.dose_policy_sha256,
            "control_policy_sha256": self.control_policy_sha256,
            "cohort_policy_sha256": self.cohort_policy_sha256,
            "family_confidence": self.family_confidence,
            "hypothesis_count": self.hypothesis_count,
            "per_hypothesis_alpha": self.per_hypothesis_alpha,
            "tasks": [asdict(task) for task in self.tasks],
            "total_layers": self.total_layers,
            "cohort_specific_layers": self.cohort_specific_layers,
            "layers": [asdict(layer) for layer in self.layers],
            "finding_codes": list(self.finding_codes),
        }

    def as_dict(self) -> dict[str, object]:
        return {**self.body(), "evidence_sha256": self.evidence_sha256}

    def with_digest(self) -> PatchControlCohortAudit:
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
        raise PatchControlCohortError("NON_CANONICAL_COHORT_EVIDENCE") from exc
    return hashlib.sha256(payload).hexdigest()


def _task_id_digest(task_id: str) -> str:
    return hashlib.sha256(b"struct-xai/patch-control-task/v1\x00" + task_id.encode("ascii")).hexdigest()


def _wilson_lower_bound(successes: int, total: int, per_hypothesis_alpha: float) -> float:
    z = NormalDist().inv_cdf(1.0 - per_hypothesis_alpha)
    observed = successes / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = (observed + z_squared / (2.0 * total)) / denominator
    spread = (
        z / denominator * math.sqrt(observed * (1.0 - observed) / total + z_squared / (4.0 * total * total))
    )
    return max(0.0, center - spread)


def audit_patch_control_cohort(
    tasks: Mapping[str, PatchControlTask],
    *,
    dose_policy: DoseResponseAuditPolicy | None = None,
    control_policy: PatchControlPolicy | None = None,
    cohort_policy: PatchControlCohortPolicy | None = None,
) -> PatchControlCohortAudit:
    """Aggregate matched-control specificity across a fixed task cohort."""

    active_dose_policy = dose_policy or DoseResponseAuditPolicy()
    active_control_policy = control_policy or PatchControlPolicy()
    active_cohort_policy = cohort_policy or PatchControlCohortPolicy()
    if not isinstance(tasks, Mapping):
        raise PatchControlCohortError("INVALID_CONTROL_COHORT")
    if not active_cohort_policy.min_tasks <= len(tasks) <= active_cohort_policy.max_tasks:
        raise PatchControlCohortError("COHORT_TASK_COUNT_OUT_OF_BOUNDS")
    if any(not isinstance(task_id, str) or _TASK_ID_RE.fullmatch(task_id) is None for task_id in tasks):
        raise PatchControlCohortError("INVALID_COHORT_TASK_ID")

    ordered_tasks = tuple(sorted(tasks.items(), key=lambda item: item[0]))
    task_evidence: list[CohortTaskEvidence] = []
    audits = []
    target_commitments: set[str] = set()
    reference_identity: tuple[object, ...] | None = None
    for task_id, task in ordered_tasks:
        if not isinstance(task, PatchControlTask):
            raise PatchControlCohortError("INVALID_CONTROL_COHORT_TASK")
        audit = audit_patch_control_specificity(
            task.treatment,
            task.controls,
            dose_policy=active_dose_policy,
            control_policy=active_control_policy,
        )
        identity = (
            audit.model_revision,
            audit.sweep_spec_sha256,
            audit.sweep_policy_sha256,
            audit.dose_policy_sha256,
            audit.control_policy_sha256,
            tuple(layer.layer for layer in audit.layers),
        )
        if reference_identity is None:
            reference_identity = identity
        elif identity != reference_identity:
            raise PatchControlCohortError("COHORT_EXPERIMENT_MISMATCH")
        if audit.target_input_sha256 in target_commitments:
            raise PatchControlCohortError("COHORT_TARGET_NOT_DISTINCT")
        target_commitments.add(audit.target_input_sha256)
        task_evidence.append(
            CohortTaskEvidence(
                task_id_sha256=_task_id_digest(task_id),
                target_input_sha256=audit.target_input_sha256,
                treatment_source_input_sha256=task.treatment.source_input_sha256,
                treatment_sweep_evidence_sha256=audit.treatment_sweep_evidence_sha256,
                control_audit_sha256=audit.evidence_sha256,
                control_count=len(audit.controls),
            )
        )
        audits.append(audit)

    assert reference_identity is not None
    layer_count = len(audits[0].layers)
    per_hypothesis_alpha = (1.0 - float(active_cohort_policy.family_confidence)) / layer_count
    layers: list[CohortLayerAssessment] = []
    for index, reference_layer in enumerate(audits[0].layers):
        matched = tuple(audit.layers[index] for audit in audits)
        if any(layer.layer != reference_layer.layer for layer in matched):
            raise PatchControlCohortError("COHORT_EXPERIMENT_MISMATCH")
        specific_tasks = sum(layer.specific for layer in matched)
        lower_bound = _wilson_lower_bound(specific_tasks, len(matched), per_hypothesis_alpha)
        findings = (
            ()
            if lower_bound >= active_cohort_policy.min_specific_task_fraction_lower_bound
            else ("SPECIFIC_TASK_FRACTION_LOWER_BOUND_BELOW_MINIMUM",)
        )
        layers.append(
            CohortLayerAssessment(
                layer=reference_layer.layer,
                specific_tasks=specific_tasks,
                total_tasks=len(matched),
                observed_specific_fraction=specific_tasks / len(matched),
                simultaneous_lower_bound=lower_bound,
                mean_endpoint_excess=fmean(layer.endpoint_excess for layer in matched),
                mean_auc_excess=fmean(layer.auc_excess for layer in matched),
                cohort_specific=not findings,
                finding_codes=findings,
            )
        )

    specific_layers = sum(layer.cohort_specific for layer in layers)
    global_findings = (
        ()
        if specific_layers >= active_cohort_policy.min_cohort_specific_layers
        else ("COHORT_SPECIFIC_LAYER_COUNT_BELOW_MINIMUM",)
    )
    result = PatchControlCohortAudit(
        schema_version=COHORT_SCHEMA_VERSION,
        status="accepted" if not global_findings else "rejected",
        bound_method=BOUND_METHOD,
        model_revision=str(reference_identity[0]),
        sweep_spec_sha256=str(reference_identity[1]),
        sweep_policy_sha256=str(reference_identity[2]),
        dose_policy_sha256=str(reference_identity[3]),
        control_policy_sha256=str(reference_identity[4]),
        cohort_policy_sha256=_digest(active_cohort_policy.as_dict()),
        family_confidence=float(active_cohort_policy.family_confidence),
        hypothesis_count=layer_count,
        per_hypothesis_alpha=per_hypothesis_alpha,
        tasks=tuple(task_evidence),
        total_layers=layer_count,
        cohort_specific_layers=specific_layers,
        layers=tuple(layers),
        finding_codes=global_findings,
        evidence_sha256="",
    )
    return result.with_digest()


def require_patch_control_cohort(
    tasks: Mapping[str, PatchControlTask],
    *,
    dose_policy: DoseResponseAuditPolicy | None = None,
    control_policy: PatchControlPolicy | None = None,
    cohort_policy: PatchControlCohortPolicy | None = None,
) -> PatchControlCohortAudit:
    """Return accepted cohort evidence or raise one stable gate error."""

    audit = audit_patch_control_cohort(
        tasks,
        dose_policy=dose_policy,
        control_policy=control_policy,
        cohort_policy=cohort_policy,
    )
    if audit.status != "accepted":
        raise PatchControlCohortRejected("PATCH_CONTROL_COHORT_REJECTED")
    return audit
