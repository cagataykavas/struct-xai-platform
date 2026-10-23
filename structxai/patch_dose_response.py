from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import struct
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from statistics import fmean
from typing import Any

import torch

from structxai.core import Candidate

REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,127}$")
MUTABLE_REVISIONS = {"head", "latest", "main", "master", "stable"}


class PatchSweepError(ValueError):
    """Invalid or operationally untrustworthy activation-patching evidence."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class DoseResponseAuditRejected(PatchSweepError):
    """A valid sweep failed its prespecified promotion policy."""


class CrossSeedDoseResponseRejected(PatchSweepError):
    """Valid per-seed sweeps failed the cross-seed promotion policy."""


@dataclass(frozen=True)
class SweepPolicy:
    max_layers: int = 96
    max_alphas: int = 16
    max_model_calls: int = 512
    max_sequence_tokens: int = 4_096
    max_hidden_width: int = 32_768
    min_margin_gap: float = 1e-4
    baseline_atol: float = 1e-6
    baseline_rtol: float = 1e-5

    def __post_init__(self) -> None:
        for name, value, lower, upper in (
            ("max_layers", self.max_layers, 1, 1_024),
            ("max_alphas", self.max_alphas, 3, 256),
            ("max_model_calls", self.max_model_calls, 4, 100_000),
            ("max_sequence_tokens", self.max_sequence_tokens, 1, 1_000_000),
            ("max_hidden_width", self.max_hidden_width, 1, 1_000_000),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"{name} must be between {lower} and {upper}")
        for name, value in (
            ("min_margin_gap", self.min_margin_gap),
            ("baseline_atol", self.baseline_atol),
            ("baseline_rtol", self.baseline_rtol),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class PatchSweepSpec:
    model_revision: str
    layer_indices: tuple[int, ...]
    alphas: tuple[float, ...]
    source_position: int
    target_position: int
    score_position: int
    positive_candidate: Candidate
    negative_candidate: Candidate

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model_revision, str)
            or not REVISION.fullmatch(self.model_revision)
            or self.model_revision.casefold() in MUTABLE_REVISIONS
        ):
            raise PatchSweepError("INVALID_MODEL_REVISION")
        if not self.layer_indices or any(
            isinstance(layer, bool) or not isinstance(layer, int) or layer < 0 for layer in self.layer_indices
        ):
            raise PatchSweepError("INVALID_LAYER_INDICES")
        if tuple(sorted(set(self.layer_indices))) != self.layer_indices:
            raise PatchSweepError("LAYER_INDICES_NOT_STRICTLY_INCREASING")
        if len(self.alphas) < 3 or any(
            isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha)
            for alpha in self.alphas
        ):
            raise PatchSweepError("INVALID_ALPHA_LADDER")
        normalized = tuple(float(alpha) for alpha in self.alphas)
        if normalized[0] != 0.0 or normalized[-1] != 1.0:
            raise PatchSweepError("ALPHA_ENDPOINTS_REQUIRED")
        if any(not 0.0 <= alpha <= 1.0 for alpha in normalized) or any(
            current <= previous for previous, current in pairwise(normalized)
        ):
            raise PatchSweepError("ALPHAS_NOT_STRICTLY_INCREASING")
        for position in (self.source_position, self.target_position, self.score_position):
            if isinstance(position, bool) or not isinstance(position, int):
                raise PatchSweepError("INVALID_TOKEN_POSITION")
        _validate_candidate(self.positive_candidate)
        _validate_candidate(self.negative_candidate)
        if self.positive_candidate.label == self.negative_candidate.label:
            raise PatchSweepError("DUPLICATE_CANDIDATE_LABEL")
        if self.positive_candidate.token_ids[0] == self.negative_candidate.token_ids[0]:
            raise PatchSweepError("DUPLICATE_CANDIDATE_TOKEN")


@dataclass(frozen=True)
class PatchSweepPoint:
    layer: int
    alpha: float
    positive_score: float
    negative_score: float
    margin: float
    margin_delta: float
    normalized_recovery: float


@dataclass(frozen=True)
class PatchSweepResult:
    schema_version: int
    model_revision: str
    source_input_sha256: str
    target_input_sha256: str
    sweep_spec_sha256: str
    policy_sha256: str
    source_margin: float
    target_margin: float
    repeated_target_margin: float
    model_calls: int
    points: tuple[PatchSweepPoint, ...]
    evidence_sha256: str

    def body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_revision": self.model_revision,
            "source_input_sha256": self.source_input_sha256,
            "target_input_sha256": self.target_input_sha256,
            "sweep_spec_sha256": self.sweep_spec_sha256,
            "policy_sha256": self.policy_sha256,
            "source_margin": self.source_margin,
            "target_margin": self.target_margin,
            "repeated_target_margin": self.repeated_target_margin,
            "model_calls": self.model_calls,
            "points": [asdict(point) for point in self.points],
        }

    def as_dict(self) -> dict[str, object]:
        return {**self.body(), "evidence_sha256": self.evidence_sha256}

    def with_digest(self) -> PatchSweepResult:
        return replace(self, evidence_sha256=_digest(self.body()))


@dataclass(frozen=True)
class DoseResponseAuditPolicy:
    """Prespecified acceptance bounds for a completed dose-response sweep."""

    min_endpoint_recovery: float = 0.10
    min_monotonic_fraction: float = 0.75
    monotonic_atol: float = 1e-6
    max_step_regression: float = 0.05
    max_abs_recovery: float = 5.0
    min_qualified_layers: int = 1

    def __post_init__(self) -> None:
        for name, value, lower, upper in (
            ("min_endpoint_recovery", self.min_endpoint_recovery, 0.0, 1.0),
            ("min_monotonic_fraction", self.min_monotonic_fraction, 0.0, 1.0),
            ("monotonic_atol", self.monotonic_atol, 0.0, 1.0),
            ("max_step_regression", self.max_step_regression, 0.0, 10.0),
            ("max_abs_recovery", self.max_abs_recovery, 1e-12, 1_000.0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not lower <= float(value) <= upper
            ):
                raise ValueError(f"{name} must be finite and between {lower} and {upper}")
        if (
            isinstance(self.min_qualified_layers, bool)
            or not isinstance(self.min_qualified_layers, int)
            or not 1 <= self.min_qualified_layers <= 1_024
        ):
            raise ValueError("min_qualified_layers must be between 1 and 1024")

    def as_dict(self) -> dict[str, int | float]:
        return {
            "min_endpoint_recovery": float(self.min_endpoint_recovery),
            "min_monotonic_fraction": float(self.min_monotonic_fraction),
            "monotonic_atol": float(self.monotonic_atol),
            "max_step_regression": float(self.max_step_regression),
            "max_abs_recovery": float(self.max_abs_recovery),
            "min_qualified_layers": self.min_qualified_layers,
        }


@dataclass(frozen=True)
class LayerDoseAssessment:
    layer: int
    endpoint_recovery: float
    normalized_auc: float
    monotonic_fraction: float
    largest_step_regression: float
    peak_abs_recovery: float
    qualified: bool
    finding_codes: tuple[str, ...]


@dataclass(frozen=True)
class DoseResponseAudit:
    schema_version: int
    status: str
    sweep_evidence_sha256: str
    sweep_spec_sha256: str
    audit_policy_sha256: str
    total_layers: int
    qualified_layers: int
    layers: tuple[LayerDoseAssessment, ...]
    finding_codes: tuple[str, ...]
    evidence_sha256: str

    def body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "sweep_evidence_sha256": self.sweep_evidence_sha256,
            "sweep_spec_sha256": self.sweep_spec_sha256,
            "audit_policy_sha256": self.audit_policy_sha256,
            "total_layers": self.total_layers,
            "qualified_layers": self.qualified_layers,
            "layers": [asdict(layer) for layer in self.layers],
            "finding_codes": list(self.finding_codes),
        }

    def as_dict(self) -> dict[str, object]:
        return {**self.body(), "evidence_sha256": self.evidence_sha256}

    def with_digest(self) -> DoseResponseAudit:
        return replace(self, evidence_sha256=_digest(self.body()))


@dataclass(frozen=True)
class CrossSeedPolicy:
    """Prespecified stability bounds across independently labelled seed runs."""

    min_seed_runs: int = 3
    max_seed_runs: int = 64
    min_seed_qualification_fraction: float = 0.80
    min_endpoint_sign_agreement: float = 0.80
    max_endpoint_range: float = 0.25
    max_auc_range: float = 0.25
    sign_atol: float = 1e-6
    min_stable_layers: int = 1

    def __post_init__(self) -> None:
        for name, value, lower, upper in (
            ("min_seed_runs", self.min_seed_runs, 2, 1_024),
            ("max_seed_runs", self.max_seed_runs, 2, 1_024),
            ("min_stable_layers", self.min_stable_layers, 1, 1_024),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"{name} must be between {lower} and {upper}")
        if self.min_seed_runs > self.max_seed_runs:
            raise ValueError("min_seed_runs cannot exceed max_seed_runs")
        for name, value, lower, upper in (
            (
                "min_seed_qualification_fraction",
                self.min_seed_qualification_fraction,
                0.0,
                1.0,
            ),
            ("min_endpoint_sign_agreement", self.min_endpoint_sign_agreement, 0.0, 1.0),
            ("max_endpoint_range", self.max_endpoint_range, 0.0, 1_000.0),
            ("max_auc_range", self.max_auc_range, 0.0, 1_000.0),
            ("sign_atol", self.sign_atol, 0.0, 1.0),
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


@dataclass(frozen=True)
class SeedSweepEvidence:
    seed: int
    sweep_evidence_sha256: str
    dose_audit_sha256: str


@dataclass(frozen=True)
class CrossSeedLayerAssessment:
    layer: int
    mean_endpoint_recovery: float
    endpoint_recovery_range: float
    mean_normalized_auc: float
    normalized_auc_range: float
    seed_qualification_fraction: float
    endpoint_sign_agreement: float
    stable: bool
    finding_codes: tuple[str, ...]


@dataclass(frozen=True)
class CrossSeedDoseResponseAudit:
    schema_version: int
    status: str
    model_revision: str
    source_input_sha256: str
    target_input_sha256: str
    sweep_spec_sha256: str
    sweep_policy_sha256: str
    dose_policy_sha256: str
    cross_seed_policy_sha256: str
    seed_runs: tuple[SeedSweepEvidence, ...]
    total_layers: int
    stable_layers: int
    layers: tuple[CrossSeedLayerAssessment, ...]
    finding_codes: tuple[str, ...]
    evidence_sha256: str

    def body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "model_revision": self.model_revision,
            "source_input_sha256": self.source_input_sha256,
            "target_input_sha256": self.target_input_sha256,
            "sweep_spec_sha256": self.sweep_spec_sha256,
            "sweep_policy_sha256": self.sweep_policy_sha256,
            "dose_policy_sha256": self.dose_policy_sha256,
            "cross_seed_policy_sha256": self.cross_seed_policy_sha256,
            "seed_runs": [asdict(run) for run in self.seed_runs],
            "total_layers": self.total_layers,
            "stable_layers": self.stable_layers,
            "layers": [asdict(layer) for layer in self.layers],
            "finding_codes": list(self.finding_codes),
        }

    def as_dict(self) -> dict[str, object]:
        return {**self.body(), "evidence_sha256": self.evidence_sha256}

    def with_digest(self) -> CrossSeedDoseResponseAudit:
        return replace(self, evidence_sha256=_digest(self.body()))


def _validate_candidate(candidate: Candidate) -> None:
    if not isinstance(candidate.label, str) or not candidate.label or len(candidate.label) > 128:
        raise PatchSweepError("INVALID_CANDIDATE_LABEL")
    if not candidate.token_ids or any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in candidate.token_ids
    ):
        raise PatchSweepError("INVALID_CANDIDATE_TOKENS")


def _digest(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise PatchSweepError("NON_CANONICAL_EVIDENCE") from exc
    return hashlib.sha256(encoded).hexdigest()


def _tensor_digest(value: torch.Tensor) -> str:
    packed = bytearray()
    for item in value.detach().to(device="cpu", dtype=torch.int64).reshape(-1).tolist():
        packed.extend(struct.pack(">q", item))
    return hashlib.sha256(bytes(packed)).hexdigest()


def _resolve_layers(model: torch.nn.Module) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "layers"):
        return model.layers
    raise PatchSweepError("UNSUPPORTED_MODEL_ARCHITECTURE")


def _validate_inputs(
    source_input_ids: torch.Tensor,
    target_input_ids: torch.Tensor,
    policy: SweepPolicy,
) -> None:
    for value in (source_input_ids, target_input_ids):
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] != 1:
            raise PatchSweepError("INVALID_INPUT_SHAPE")
        if value.dtype not in (torch.int32, torch.int64, torch.long):
            raise PatchSweepError("INVALID_INPUT_DTYPE")
        if value.shape[1] < 1 or value.shape[1] > policy.max_sequence_tokens:
            raise PatchSweepError("SEQUENCE_TOKEN_BUDGET_EXCEEDED")
        if bool(torch.any(value < 0).item()):
            raise PatchSweepError("INVALID_INPUT_TOKEN")
    if source_input_ids.device != target_input_ids.device:
        raise PatchSweepError("INPUT_DEVICE_MISMATCH")


def _resolve_position(position: int, length: int, code: str) -> int:
    resolved = position if position >= 0 else length + position
    if not 0 <= resolved < length:
        raise PatchSweepError(code)
    return resolved


def _hidden(output: object) -> torch.Tensor:
    hidden = output[0] if isinstance(output, tuple) else output
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[0] != 1:
        raise PatchSweepError("INVALID_LAYER_OUTPUT")
    if not bool(torch.isfinite(hidden).all().item()):
        raise PatchSweepError("NON_FINITE_LAYER_OUTPUT")
    return hidden


def _logits(output: object) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[0] != 1:
        raise PatchSweepError("INVALID_MODEL_OUTPUT")
    if not bool(torch.isfinite(logits).all().item()):
        raise PatchSweepError("NON_FINITE_MODEL_OUTPUT")
    return logits


def _margin(
    logits: torch.Tensor,
    position: int,
    positive_token: int,
    negative_token: int,
) -> tuple[float, float, float]:
    vocab = logits.shape[-1]
    if positive_token >= vocab or negative_token >= vocab:
        raise PatchSweepError("CANDIDATE_TOKEN_OUT_OF_RANGE")
    positive = float(logits[0, position, positive_token].item())
    negative = float(logits[0, position, negative_token].item())
    margin = positive - negative
    if not all(math.isfinite(value) for value in (positive, negative, margin)):
        raise PatchSweepError("NON_FINITE_CANDIDATE_SCORE")
    return positive, negative, margin


def _validated_layer_curves(result: PatchSweepResult) -> tuple[tuple[int, tuple[PatchSweepPoint, ...]], ...]:
    if (
        not isinstance(result, PatchSweepResult)
        or type(result.schema_version) is not int
        or result.schema_version != 1
    ):
        raise PatchSweepError("UNSUPPORTED_SWEEP_EVIDENCE")
    if not re.fullmatch(r"[0-9a-f]{64}", result.evidence_sha256) or not hmac.compare_digest(
        result.evidence_sha256, _digest(result.body())
    ):
        raise PatchSweepError("SWEEP_EVIDENCE_DIGEST_MISMATCH")
    if not result.points:
        raise PatchSweepError("EMPTY_SWEEP_EVIDENCE")
    if (
        type(result.model_calls) is not int
        or result.model_calls != 3 + len(result.points)
        or not isinstance(result.model_revision, str)
        or not REVISION.fullmatch(result.model_revision)
        or result.model_revision.casefold() in MUTABLE_REVISIONS
        or any(
            not re.fullmatch(r"[0-9a-f]{64}", digest)
            for digest in (
                result.source_input_sha256,
                result.target_input_sha256,
                result.sweep_spec_sha256,
                result.policy_sha256,
            )
        )
    ):
        raise PatchSweepError("INVALID_SWEEP_METADATA")
    if not all(
        math.isfinite(value)
        for value in (result.source_margin, result.target_margin, result.repeated_target_margin)
    ):
        raise PatchSweepError("NON_FINITE_SWEEP_EVIDENCE")

    margin_gap = result.source_margin - result.target_margin
    if margin_gap == 0.0 or not math.isfinite(margin_gap):
        raise PatchSweepError("INVALID_SWEEP_MARGIN_GAP")

    grouped: list[tuple[int, list[PatchSweepPoint]]] = []
    for point in result.points:
        if not isinstance(point, PatchSweepPoint):
            raise PatchSweepError("INVALID_SWEEP_POINT")
        if isinstance(point.layer, bool) or not isinstance(point.layer, int) or point.layer < 0:
            raise PatchSweepError("INVALID_SWEEP_LAYER")
        values = (
            point.alpha,
            point.positive_score,
            point.negative_score,
            point.margin,
            point.margin_delta,
            point.normalized_recovery,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in values
        ):
            raise PatchSweepError("NON_FINITE_SWEEP_POINT")
        if not math.isclose(point.margin, point.positive_score - point.negative_score, abs_tol=1e-9):
            raise PatchSweepError("INCONSISTENT_SWEEP_MARGIN")
        if not math.isclose(point.margin_delta, point.margin - result.target_margin, abs_tol=1e-9):
            raise PatchSweepError("INCONSISTENT_SWEEP_DELTA")
        if not math.isclose(point.normalized_recovery, point.margin_delta / margin_gap, abs_tol=1e-9):
            raise PatchSweepError("INCONSISTENT_SWEEP_RECOVERY")

        if not grouped or grouped[-1][0] != point.layer:
            if grouped and point.layer <= grouped[-1][0]:
                raise PatchSweepError("NON_CANONICAL_SWEEP_ORDER")
            grouped.append((point.layer, []))
        grouped[-1][1].append(point)

    reference_alphas: tuple[float, ...] | None = None
    curves: list[tuple[int, tuple[PatchSweepPoint, ...]]] = []
    for layer, points in grouped:
        alphas = tuple(float(point.alpha) for point in points)
        if len(alphas) < 3 or alphas[0] != 0.0 or alphas[-1] != 1.0:
            raise PatchSweepError("INCOMPLETE_ALPHA_LADDER")
        if any(current <= previous for previous, current in pairwise(alphas)):
            raise PatchSweepError("NON_CANONICAL_ALPHA_ORDER")
        if reference_alphas is None:
            reference_alphas = alphas
        elif alphas != reference_alphas:
            raise PatchSweepError("INCONSISTENT_ALPHA_LADDERS")
        if not math.isclose(points[0].normalized_recovery, 0.0, abs_tol=1e-9):
            raise PatchSweepError("INVALID_ZERO_DOSE_RECOVERY")
        curves.append((layer, tuple(points)))
    return tuple(curves)


def audit_dose_response(
    result: PatchSweepResult,
    *,
    policy: DoseResponseAuditPolicy | None = None,
) -> DoseResponseAudit:
    """Evaluate every declared layer against a prespecified dose-response policy.

    Scientific rejection is returned as a signed-by-digest audit artifact. Malformed or
    tampered sweep evidence raises :class:`PatchSweepError` instead.
    """

    active_policy = policy or DoseResponseAuditPolicy()
    curves = _validated_layer_curves(result)
    assessments: list[LayerDoseAssessment] = []

    for layer, points in curves:
        recoveries = tuple(float(point.normalized_recovery) for point in points)
        steps = tuple(current - previous for previous, current in pairwise(recoveries))
        endpoint_recovery = recoveries[-1]
        monotonic_fraction = sum(step >= -float(active_policy.monotonic_atol) for step in steps) / len(steps)
        largest_step_regression = max(0.0, -min(steps))
        peak_abs_recovery = max(abs(value) for value in recoveries)
        normalized_auc = sum(
            (points[index].alpha - points[index - 1].alpha)
            * (recoveries[index - 1] + recoveries[index])
            / 2.0
            for index in range(1, len(points))
        )
        if not all(
            math.isfinite(value)
            for value in (
                endpoint_recovery,
                monotonic_fraction,
                largest_step_regression,
                peak_abs_recovery,
                normalized_auc,
            )
        ):
            raise PatchSweepError("NON_FINITE_AUDIT_METRIC")

        findings: list[str] = []
        if endpoint_recovery < active_policy.min_endpoint_recovery:
            findings.append("ENDPOINT_RECOVERY_BELOW_MINIMUM")
        if monotonic_fraction < active_policy.min_monotonic_fraction:
            findings.append("MONOTONIC_FRACTION_BELOW_MINIMUM")
        if largest_step_regression > active_policy.max_step_regression:
            findings.append("STEP_REGRESSION_EXCEEDED")
        if peak_abs_recovery > active_policy.max_abs_recovery:
            findings.append("RECOVERY_BOUND_EXCEEDED")

        assessments.append(
            LayerDoseAssessment(
                layer=layer,
                endpoint_recovery=endpoint_recovery,
                normalized_auc=normalized_auc,
                monotonic_fraction=monotonic_fraction,
                largest_step_regression=largest_step_regression,
                peak_abs_recovery=peak_abs_recovery,
                qualified=not findings,
                finding_codes=tuple(findings),
            )
        )

    qualified_layers = sum(assessment.qualified for assessment in assessments)
    global_findings = (
        ()
        if qualified_layers >= active_policy.min_qualified_layers
        else ("QUALIFYING_LAYER_COUNT_BELOW_MINIMUM",)
    )
    audit = DoseResponseAudit(
        schema_version=1,
        status="accepted" if not global_findings else "rejected",
        sweep_evidence_sha256=result.evidence_sha256,
        sweep_spec_sha256=result.sweep_spec_sha256,
        audit_policy_sha256=_digest(active_policy.as_dict()),
        total_layers=len(assessments),
        qualified_layers=qualified_layers,
        layers=tuple(assessments),
        finding_codes=global_findings,
        evidence_sha256="",
    )
    return audit.with_digest()


def require_dose_response_acceptance(
    result: PatchSweepResult,
    *,
    policy: DoseResponseAuditPolicy | None = None,
) -> DoseResponseAudit:
    """Return an accepted audit or raise a stable promotion-gate error."""

    audit = audit_dose_response(result, policy=policy)
    if audit.status != "accepted":
        raise DoseResponseAuditRejected("DOSE_RESPONSE_AUDIT_REJECTED")
    return audit


def _effect_sign(value: float, atol: float) -> int:
    if value > atol:
        return 1
    if value < -atol:
        return -1
    return 0


def audit_cross_seed_dose_response(
    runs: Mapping[int, PatchSweepResult],
    *,
    dose_policy: DoseResponseAuditPolicy | None = None,
    cross_seed_policy: CrossSeedPolicy | None = None,
) -> CrossSeedDoseResponseAudit:
    """Aggregate matching dose-response sweeps under a prespecified seed policy."""

    active_dose_policy = dose_policy or DoseResponseAuditPolicy()
    active_cross_seed_policy = cross_seed_policy or CrossSeedPolicy()
    if not isinstance(runs, Mapping):
        raise PatchSweepError("INVALID_SEED_RUNS")
    seed_runs = tuple(runs.items())
    if not active_cross_seed_policy.min_seed_runs <= len(seed_runs) <= active_cross_seed_policy.max_seed_runs:
        raise PatchSweepError("SEED_RUN_COUNT_OUT_OF_BOUNDS")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed, _ in seed_runs):
        raise PatchSweepError("INVALID_SEED_ID")
    ordered_runs = tuple(sorted(seed_runs, key=lambda item: item[0]))

    reference_identity: tuple[object, ...] | None = None
    reference_curve_shape: tuple[tuple[int, tuple[float, ...]], ...] | None = None
    seed_evidence: list[SeedSweepEvidence] = []
    per_seed_assessments: list[tuple[int, tuple[LayerDoseAssessment, ...]]] = []

    for seed, result in ordered_runs:
        audit = audit_dose_response(result, policy=active_dose_policy)
        curves = _validated_layer_curves(result)
        identity = (
            result.model_revision,
            result.source_input_sha256,
            result.target_input_sha256,
            result.sweep_spec_sha256,
            result.policy_sha256,
        )
        curve_shape = tuple(
            (layer, tuple(float(point.alpha) for point in points)) for layer, points in curves
        )
        if reference_identity is None:
            reference_identity = identity
            reference_curve_shape = curve_shape
        elif identity != reference_identity or curve_shape != reference_curve_shape:
            raise PatchSweepError("CROSS_SEED_EXPERIMENT_MISMATCH")
        seed_evidence.append(
            SeedSweepEvidence(
                seed=seed,
                sweep_evidence_sha256=result.evidence_sha256,
                dose_audit_sha256=audit.evidence_sha256,
            )
        )
        per_seed_assessments.append((seed, audit.layers))

    if reference_identity is None or reference_curve_shape is None:
        raise PatchSweepError("SEED_RUN_COUNT_OUT_OF_BOUNDS")
    layer_assessments: list[CrossSeedLayerAssessment] = []
    for layer_index, (layer, _) in enumerate(reference_curve_shape):
        assessments = tuple(rows[layer_index] for _, rows in per_seed_assessments)
        if any(assessment.layer != layer for assessment in assessments):
            raise PatchSweepError("CROSS_SEED_EXPERIMENT_MISMATCH")
        endpoints = tuple(assessment.endpoint_recovery for assessment in assessments)
        aucs = tuple(assessment.normalized_auc for assessment in assessments)
        qualification_fraction = sum(assessment.qualified for assessment in assessments) / len(assessments)
        signs = tuple(_effect_sign(value, float(active_cross_seed_policy.sign_atol)) for value in endpoints)
        sign_agreement = max(signs.count(sign) for sign in set(signs)) / len(signs)
        endpoint_range = max(endpoints) - min(endpoints)
        auc_range = max(aucs) - min(aucs)
        findings: list[str] = []
        if qualification_fraction < active_cross_seed_policy.min_seed_qualification_fraction:
            findings.append("SEED_QUALIFICATION_FRACTION_BELOW_MINIMUM")
        if sign_agreement < active_cross_seed_policy.min_endpoint_sign_agreement:
            findings.append("ENDPOINT_SIGN_AGREEMENT_BELOW_MINIMUM")
        if endpoint_range > active_cross_seed_policy.max_endpoint_range:
            findings.append("ENDPOINT_RANGE_EXCEEDED")
        if auc_range > active_cross_seed_policy.max_auc_range:
            findings.append("AUC_RANGE_EXCEEDED")
        layer_assessments.append(
            CrossSeedLayerAssessment(
                layer=layer,
                mean_endpoint_recovery=fmean(endpoints),
                endpoint_recovery_range=endpoint_range,
                mean_normalized_auc=fmean(aucs),
                normalized_auc_range=auc_range,
                seed_qualification_fraction=qualification_fraction,
                endpoint_sign_agreement=sign_agreement,
                stable=not findings,
                finding_codes=tuple(findings),
            )
        )

    stable_layers = sum(assessment.stable for assessment in layer_assessments)
    global_findings = (
        ()
        if stable_layers >= active_cross_seed_policy.min_stable_layers
        else ("STABLE_LAYER_COUNT_BELOW_MINIMUM",)
    )
    result = CrossSeedDoseResponseAudit(
        schema_version=1,
        status="accepted" if not global_findings else "rejected",
        model_revision=str(reference_identity[0]),
        source_input_sha256=str(reference_identity[1]),
        target_input_sha256=str(reference_identity[2]),
        sweep_spec_sha256=str(reference_identity[3]),
        sweep_policy_sha256=str(reference_identity[4]),
        dose_policy_sha256=_digest(active_dose_policy.as_dict()),
        cross_seed_policy_sha256=_digest(active_cross_seed_policy.as_dict()),
        seed_runs=tuple(seed_evidence),
        total_layers=len(layer_assessments),
        stable_layers=stable_layers,
        layers=tuple(layer_assessments),
        finding_codes=global_findings,
        evidence_sha256="",
    )
    return result.with_digest()


def require_cross_seed_dose_response_acceptance(
    runs: Mapping[int, PatchSweepResult],
    *,
    dose_policy: DoseResponseAuditPolicy | None = None,
    cross_seed_policy: CrossSeedPolicy | None = None,
) -> CrossSeedDoseResponseAudit:
    """Return accepted cross-seed evidence or raise a stable gate error."""

    audit = audit_cross_seed_dose_response(
        runs,
        dose_policy=dose_policy,
        cross_seed_policy=cross_seed_policy,
    )
    if audit.status != "accepted":
        raise CrossSeedDoseResponseRejected("CROSS_SEED_DOSE_RESPONSE_REJECTED")
    return audit


def run_residual_patch_sweep(
    model: torch.nn.Module,
    *,
    source_input_ids: torch.Tensor,
    target_input_ids: torch.Tensor,
    spec: PatchSweepSpec,
    policy: SweepPolicy | None = None,
) -> PatchSweepResult:
    """Run a bounded residual-stream interpolation sweep on a preloaded causal LM.

    The source and unpatched target activations are captured once. At each governed layer,
    the target position is replaced by ``target + alpha * (source - target)``. Candidate
    evidence uses the same explicit first-token logit margin as the existing platform.
    """

    active_policy = policy or SweepPolicy()
    _validate_inputs(source_input_ids, target_input_ids, active_policy)
    if len(spec.layer_indices) > active_policy.max_layers:
        raise PatchSweepError("LAYER_BUDGET_EXCEEDED")
    if len(spec.alphas) > active_policy.max_alphas:
        raise PatchSweepError("ALPHA_BUDGET_EXCEEDED")
    expected_calls = 3 + len(spec.layer_indices) * len(spec.alphas)
    if expected_calls > active_policy.max_model_calls:
        raise PatchSweepError("MODEL_CALL_BUDGET_EXCEEDED")

    layers = _resolve_layers(model)
    if spec.layer_indices[-1] >= len(layers):
        raise PatchSweepError("LAYER_INDEX_OUT_OF_RANGE")
    source_position = _resolve_position(
        spec.source_position, source_input_ids.shape[1], "SOURCE_POSITION_OUT_OF_RANGE"
    )
    target_position = _resolve_position(
        spec.target_position, target_input_ids.shape[1], "TARGET_POSITION_OUT_OF_RANGE"
    )
    score_position = _resolve_position(
        spec.score_position, target_input_ids.shape[1], "SCORE_POSITION_OUT_OF_RANGE"
    )
    source_score_position = _resolve_position(
        spec.score_position, source_input_ids.shape[1], "SOURCE_SCORE_POSITION_OUT_OF_RANGE"
    )

    was_training = model.training
    source_activations: dict[int, torch.Tensor] = {}
    target_activations: dict[int, torch.Tensor] = {}
    positive_token = spec.positive_candidate.token_ids[0]
    negative_token = spec.negative_candidate.token_ids[0]

    def capture(store: dict[int, torch.Tensor], layer: int, position: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            hidden = _hidden(output)
            if hidden.shape[-1] > active_policy.max_hidden_width:
                raise PatchSweepError("HIDDEN_WIDTH_BUDGET_EXCEEDED")
            store[layer] = hidden[:, position, :].detach().clone()

        return hook

    def forward_with_capture(
        input_ids: torch.Tensor,
        store: dict[int, torch.Tensor],
        position: int,
    ) -> object:
        handles = [
            layers[layer].register_forward_hook(capture(store, layer, position))
            for layer in spec.layer_indices
        ]
        try:
            with torch.no_grad():
                return model(input_ids=input_ids, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

    def make_patch_hook(
        source_state: torch.Tensor,
        target_state: torch.Tensor,
        alpha: float,
    ):
        def patch_hook(
            _module: torch.nn.Module,
            _inputs: tuple[object, ...],
            output: object,
        ) -> object:
            hidden = _hidden(output)
            replacement = target_state + alpha * (source_state - target_state)
            replacement = replacement.to(device=hidden.device, dtype=hidden.dtype)
            patched = hidden.clone()
            patched[:, target_position, :] = replacement
            if isinstance(output, tuple):
                return (patched, *output[1:])
            return patched

        return patch_hook

    model.eval()
    try:
        source_output = forward_with_capture(source_input_ids, source_activations, source_position)
        target_output = forward_with_capture(target_input_ids, target_activations, target_position)
        if set(source_activations) != set(spec.layer_indices) or set(target_activations) != set(
            spec.layer_indices
        ):
            raise PatchSweepError("INCOMPLETE_ACTIVATION_CAPTURE")
        for layer in spec.layer_indices:
            if source_activations[layer].shape != target_activations[layer].shape:
                raise PatchSweepError("ACTIVATION_SHAPE_MISMATCH")

        _, _, source_margin = _margin(
            _logits(source_output), source_score_position, positive_token, negative_token
        )
        _, _, target_margin = _margin(_logits(target_output), score_position, positive_token, negative_token)
        margin_gap = source_margin - target_margin
        if abs(margin_gap) < active_policy.min_margin_gap:
            raise PatchSweepError("WEAK_SOURCE_TARGET_MARGIN_GAP")

        points: list[PatchSweepPoint] = []
        for layer in spec.layer_indices:
            source_state = source_activations[layer]
            target_state = target_activations[layer]
            for raw_alpha in spec.alphas:
                alpha = float(raw_alpha)
                handle = layers[layer].register_forward_hook(
                    make_patch_hook(source_state, target_state, alpha)
                )
                try:
                    with torch.no_grad():
                        patched_output = model(input_ids=target_input_ids, use_cache=False)
                finally:
                    handle.remove()
                positive, negative, margin = _margin(
                    _logits(patched_output), score_position, positive_token, negative_token
                )
                if alpha == 0.0 and not math.isclose(
                    margin,
                    target_margin,
                    abs_tol=active_policy.baseline_atol,
                    rel_tol=active_policy.baseline_rtol,
                ):
                    raise PatchSweepError("ZERO_DOSE_BASELINE_MISMATCH")
                points.append(
                    PatchSweepPoint(
                        layer=layer,
                        alpha=alpha,
                        positive_score=positive,
                        negative_score=negative,
                        margin=margin,
                        margin_delta=margin - target_margin,
                        normalized_recovery=(margin - target_margin) / margin_gap,
                    )
                )

        with torch.no_grad():
            repeated_output = model(input_ids=target_input_ids, use_cache=False)
        _, _, repeated_target_margin = _margin(
            _logits(repeated_output), score_position, positive_token, negative_token
        )
        if not math.isclose(
            repeated_target_margin,
            target_margin,
            abs_tol=active_policy.baseline_atol,
            rel_tol=active_policy.baseline_rtol,
        ):
            raise PatchSweepError("TARGET_BASELINE_DRIFT")
    finally:
        model.train(was_training)

    report = PatchSweepResult(
        schema_version=1,
        model_revision=spec.model_revision,
        source_input_sha256=_tensor_digest(source_input_ids),
        target_input_sha256=_tensor_digest(target_input_ids),
        sweep_spec_sha256=_digest(
            {
                "model_revision": spec.model_revision,
                "layer_indices": list(spec.layer_indices),
                "alphas": [float(alpha) for alpha in spec.alphas],
                "source_position": source_position,
                "target_position": target_position,
                "source_score_position": source_score_position,
                "target_score_position": score_position,
                "positive_token_id": positive_token,
                "negative_token_id": negative_token,
            }
        ),
        policy_sha256=_digest(active_policy.as_dict()),
        source_margin=source_margin,
        target_margin=target_margin,
        repeated_target_margin=repeated_target_margin,
        model_calls=expected_calls,
        points=tuple(points),
        evidence_sha256="",
    )
    return report.with_digest()
