from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
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
