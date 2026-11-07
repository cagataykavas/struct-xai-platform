from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace

from structxai.patch_dose_response import (
    DoseResponseAuditPolicy,
    PatchSweepError,
    PatchSweepResult,
    audit_dose_response,
)

CONTROL_SPECIFICITY_SCHEMA_VERSION = 1
_CONTROL_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


class PatchControlAuditError(PatchSweepError):
    """Matched controls are malformed or incompatible with the treatment sweep."""


class PatchControlAuditRejected(PatchControlAuditError):
    """Valid matched-control evidence failed its prespecified specificity policy."""


@dataclass(frozen=True, slots=True)
class PatchControlPolicy:
    min_controls: int = 3
    max_controls: int = 32
    min_endpoint_excess: float = 0.15
    min_auc_excess: float = 0.10
    max_control_endpoint_recovery: float = 0.25
    max_control_normalized_auc: float = 0.20
    min_specific_layers: int = 1

    def __post_init__(self) -> None:
        for name, value, lower, upper in (
            ("min_controls", self.min_controls, 2, 64),
            ("max_controls", self.max_controls, 2, 64),
            ("min_specific_layers", self.min_specific_layers, 1, 1_024),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"{name} must be between {lower} and {upper}")
        if self.min_controls > self.max_controls:
            raise ValueError("min_controls cannot exceed max_controls")
        for name, value in (
            ("min_endpoint_excess", self.min_endpoint_excess),
            ("min_auc_excess", self.min_auc_excess),
            ("max_control_endpoint_recovery", self.max_control_endpoint_recovery),
            ("max_control_normalized_auc", self.max_control_normalized_auc),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= float(value) <= 1_000.0
            ):
                raise ValueError(f"{name} must be finite and between 0 and 1000")

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ControlSweepEvidence:
    control_id_sha256: str
    source_input_sha256: str
    sweep_evidence_sha256: str
    dose_audit_sha256: str


@dataclass(frozen=True, slots=True)
class LayerControlAssessment:
    layer: int
    treatment_endpoint_recovery: float
    strongest_control_endpoint_recovery: float
    endpoint_excess: float
    treatment_normalized_auc: float
    strongest_control_normalized_auc: float
    auc_excess: float
    treatment_qualified: bool
    specific: bool
    finding_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PatchControlAudit:
    schema_version: int
    status: str
    model_revision: str
    target_input_sha256: str
    sweep_spec_sha256: str
    sweep_policy_sha256: str
    treatment_sweep_evidence_sha256: str
    treatment_dose_audit_sha256: str
    dose_policy_sha256: str
    control_policy_sha256: str
    controls: tuple[ControlSweepEvidence, ...]
    total_layers: int
    specific_layers: int
    layers: tuple[LayerControlAssessment, ...]
    finding_codes: tuple[str, ...]
    evidence_sha256: str

    def body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "model_revision": self.model_revision,
            "target_input_sha256": self.target_input_sha256,
            "sweep_spec_sha256": self.sweep_spec_sha256,
            "sweep_policy_sha256": self.sweep_policy_sha256,
            "treatment_sweep_evidence_sha256": self.treatment_sweep_evidence_sha256,
            "treatment_dose_audit_sha256": self.treatment_dose_audit_sha256,
            "dose_policy_sha256": self.dose_policy_sha256,
            "control_policy_sha256": self.control_policy_sha256,
            "controls": [asdict(control) for control in self.controls],
            "total_layers": self.total_layers,
            "specific_layers": self.specific_layers,
            "layers": [asdict(layer) for layer in self.layers],
            "finding_codes": list(self.finding_codes),
        }

    def as_dict(self) -> dict[str, object]:
        return {**self.body(), "evidence_sha256": self.evidence_sha256}

    def with_digest(self) -> PatchControlAudit:
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
        raise PatchControlAuditError("NON_CANONICAL_CONTROL_EVIDENCE") from exc
    return hashlib.sha256(payload).hexdigest()


def _control_id_digest(control_id: str) -> str:
    return hashlib.sha256(b"struct-xai/patch-control-id/v1\x00" + control_id.encode("ascii")).hexdigest()


def audit_patch_control_specificity(
    treatment: PatchSweepResult,
    controls: Mapping[str, PatchSweepResult],
    *,
    dose_policy: DoseResponseAuditPolicy | None = None,
    control_policy: PatchControlPolicy | None = None,
) -> PatchControlAudit:
    """Require a treatment dose response to exceed every matched null sweep."""

    active_dose_policy = dose_policy or DoseResponseAuditPolicy()
    active_control_policy = control_policy or PatchControlPolicy()
    if not isinstance(controls, Mapping):
        raise PatchControlAuditError("INVALID_CONTROL_SWEEPS")
    if not active_control_policy.min_controls <= len(controls) <= active_control_policy.max_controls:
        raise PatchControlAuditError("CONTROL_COUNT_OUT_OF_BOUNDS")
    if any(not isinstance(key, str) or _CONTROL_ID_RE.fullmatch(key) is None for key in controls):
        raise PatchControlAuditError("INVALID_CONTROL_ID")

    treatment_audit = audit_dose_response(treatment, policy=active_dose_policy)
    treatment_identity = (
        treatment.model_revision,
        treatment.target_input_sha256,
        treatment.sweep_spec_sha256,
        treatment.policy_sha256,
    )
    treatment_layers = tuple(layer.layer for layer in treatment_audit.layers)
    ordered_controls = tuple(sorted(controls.items(), key=lambda item: item[0]))
    source_commitments = {treatment.source_input_sha256}
    control_evidence: list[ControlSweepEvidence] = []
    control_assessments = []
    for control_id, control in ordered_controls:
        if not isinstance(control, PatchSweepResult):
            raise PatchControlAuditError("INVALID_CONTROL_SWEEP")
        control_audit = audit_dose_response(control, policy=active_dose_policy)
        identity = (
            control.model_revision,
            control.target_input_sha256,
            control.sweep_spec_sha256,
            control.policy_sha256,
        )
        if (
            identity != treatment_identity
            or tuple(layer.layer for layer in control_audit.layers) != treatment_layers
        ):
            raise PatchControlAuditError("CONTROL_EXPERIMENT_MISMATCH")
        if control.source_input_sha256 in source_commitments:
            raise PatchControlAuditError("CONTROL_SOURCE_NOT_DISTINCT")
        source_commitments.add(control.source_input_sha256)
        control_evidence.append(
            ControlSweepEvidence(
                control_id_sha256=_control_id_digest(control_id),
                source_input_sha256=control.source_input_sha256,
                sweep_evidence_sha256=control.evidence_sha256,
                dose_audit_sha256=control_audit.evidence_sha256,
            )
        )
        control_assessments.append(control_audit.layers)

    layers: list[LayerControlAssessment] = []
    for index, treatment_layer in enumerate(treatment_audit.layers):
        matched_controls = tuple(rows[index] for rows in control_assessments)
        if any(row.layer != treatment_layer.layer for row in matched_controls):
            raise PatchControlAuditError("CONTROL_EXPERIMENT_MISMATCH")
        strongest_endpoint = max(row.endpoint_recovery for row in matched_controls)
        strongest_auc = max(row.normalized_auc for row in matched_controls)
        endpoint_excess = treatment_layer.endpoint_recovery - strongest_endpoint
        auc_excess = treatment_layer.normalized_auc - strongest_auc
        findings: list[str] = []
        if not treatment_layer.qualified:
            findings.append("TREATMENT_LAYER_NOT_QUALIFIED")
        if strongest_endpoint > active_control_policy.max_control_endpoint_recovery:
            findings.append("CONTROL_ENDPOINT_CEILING_EXCEEDED")
        if strongest_auc > active_control_policy.max_control_normalized_auc:
            findings.append("CONTROL_AUC_CEILING_EXCEEDED")
        if endpoint_excess < active_control_policy.min_endpoint_excess:
            findings.append("ENDPOINT_EXCESS_BELOW_MINIMUM")
        if auc_excess < active_control_policy.min_auc_excess:
            findings.append("AUC_EXCESS_BELOW_MINIMUM")
        layers.append(
            LayerControlAssessment(
                layer=treatment_layer.layer,
                treatment_endpoint_recovery=treatment_layer.endpoint_recovery,
                strongest_control_endpoint_recovery=strongest_endpoint,
                endpoint_excess=endpoint_excess,
                treatment_normalized_auc=treatment_layer.normalized_auc,
                strongest_control_normalized_auc=strongest_auc,
                auc_excess=auc_excess,
                treatment_qualified=treatment_layer.qualified,
                specific=not findings,
                finding_codes=tuple(findings),
            )
        )

    specific_layers = sum(layer.specific for layer in layers)
    global_findings: list[str] = []
    if treatment_audit.status != "accepted":
        global_findings.append("TREATMENT_DOSE_RESPONSE_REJECTED")
    if specific_layers < active_control_policy.min_specific_layers:
        global_findings.append("SPECIFIC_LAYER_COUNT_BELOW_MINIMUM")
    result = PatchControlAudit(
        schema_version=CONTROL_SPECIFICITY_SCHEMA_VERSION,
        status="accepted" if not global_findings else "rejected",
        model_revision=treatment.model_revision,
        target_input_sha256=treatment.target_input_sha256,
        sweep_spec_sha256=treatment.sweep_spec_sha256,
        sweep_policy_sha256=treatment.policy_sha256,
        treatment_sweep_evidence_sha256=treatment.evidence_sha256,
        treatment_dose_audit_sha256=treatment_audit.evidence_sha256,
        dose_policy_sha256=treatment_audit.audit_policy_sha256,
        control_policy_sha256=_digest(active_control_policy.as_dict()),
        controls=tuple(control_evidence),
        total_layers=len(layers),
        specific_layers=specific_layers,
        layers=tuple(layers),
        finding_codes=tuple(global_findings),
        evidence_sha256="",
    )
    return result.with_digest()


def require_patch_control_specificity(
    treatment: PatchSweepResult,
    controls: Mapping[str, PatchSweepResult],
    *,
    dose_policy: DoseResponseAuditPolicy | None = None,
    control_policy: PatchControlPolicy | None = None,
) -> PatchControlAudit:
    """Return accepted specificity evidence or raise one stable gate error."""

    audit = audit_patch_control_specificity(
        treatment,
        controls,
        dose_policy=dose_policy,
        control_policy=control_policy,
    )
    if audit.status != "accepted":
        raise PatchControlAuditRejected("PATCH_CONTROL_SPECIFICITY_REJECTED")
    return audit
