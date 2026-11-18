from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

MAX_ATTRIBUTION_TOKENS = 8_192
MAX_HIDDEN_DIMENSION = 65_536
MAX_EMBEDDING_ELEMENTS = 16_777_216


@dataclass(frozen=True)
class TokenAttribution:
    position: int
    token_id: int
    attribution: float
    absolute_share: float


@dataclass(frozen=True)
class MarginAttribution:
    method: str
    reference: str
    positive_candidate: str
    negative_candidate: str
    margin: float
    attribution_sum: float
    absolute_attribution_sum: float
    completeness_residual: float
    tokens: tuple[TokenAttribution, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_scalar_score(name: str, score: torch.Tensor, device: torch.device) -> None:
    if not isinstance(score, torch.Tensor) or score.numel() != 1:
        raise TypeError(f"{name} must be a scalar tensor")
    if not score.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if score.device != device:
        raise ValueError(f"{name} must be on the same device as token_embeddings")
    if not bool(torch.isfinite(score.detach()).item()):
        raise ValueError(f"{name} must be finite")


def gradient_input_margin_attribution(
    token_embeddings: torch.Tensor,
    positive_score: torch.Tensor,
    negative_score: torch.Tensor,
    token_ids: tuple[int, ...] | list[int],
    *,
    positive_candidate: str,
    negative_candidate: str,
) -> MarginAttribution:
    """Attribute a signed candidate margin to prompt-token embeddings.

    The result is the dot product of each token embedding and the gradient of
    ``positive_score - negative_score`` with respect to that embedding. It is a
    local first-order diagnostic around the observed prompt, not an integrated
    path attribution or a causal effect estimate.
    """
    if token_embeddings.ndim != 2:
        raise ValueError("token_embeddings must have shape [tokens, hidden_dimension]")
    if not token_embeddings.is_floating_point():
        raise TypeError("token_embeddings must use a floating-point dtype")
    if not token_embeddings.requires_grad:
        raise ValueError("token_embeddings must require gradients")

    token_count, hidden_dimension = token_embeddings.shape
    if not 1 <= token_count <= MAX_ATTRIBUTION_TOKENS:
        raise ValueError(f"token count must be between 1 and {MAX_ATTRIBUTION_TOKENS}")
    if not 1 <= hidden_dimension <= MAX_HIDDEN_DIMENSION:
        raise ValueError(f"hidden dimension must be between 1 and {MAX_HIDDEN_DIMENSION}")
    if token_embeddings.numel() > MAX_EMBEDDING_ELEMENTS:
        raise ValueError(f"token embedding work exceeds {MAX_EMBEDDING_ELEMENTS} elements")
    if not bool(torch.isfinite(token_embeddings.detach()).all().item()):
        raise ValueError("token_embeddings must be finite")

    ids = tuple(token_ids)
    if len(ids) != token_count:
        raise ValueError("token_ids must contain one ID per token embedding")
    if any(isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0 for token_id in ids):
        raise ValueError("token_ids must be non-negative integers")
    if not positive_candidate.strip() or not negative_candidate.strip():
        raise ValueError("candidate labels must be non-empty")
    if positive_candidate == negative_candidate:
        raise ValueError("candidate labels must be distinct")

    _validate_scalar_score("positive_score", positive_score, token_embeddings.device)
    _validate_scalar_score("negative_score", negative_score, token_embeddings.device)
    margin = positive_score.reshape(()) - negative_score.reshape(())
    try:
        (gradient,) = torch.autograd.grad(
            margin,
            token_embeddings,
            allow_unused=True,
            create_graph=False,
            retain_graph=False,
        )
    except RuntimeError as exc:
        raise ValueError("candidate margin gradient could not be computed") from exc
    if gradient is None:
        raise ValueError("candidate margin is disconnected from token_embeddings")
    if not bool(torch.isfinite(gradient.detach()).all().item()):
        raise ValueError("candidate margin gradient must be finite")

    raw = (gradient * token_embeddings).sum(dim=-1).detach().to(dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(raw).all().item()):
        raise ValueError("token attributions must be finite")
    absolute_total = float(raw.abs().sum().item())
    rows = tuple(
        TokenAttribution(
            position=position,
            token_id=token_id,
            attribution=float(value.item()),
            absolute_share=(float(value.abs().item()) / absolute_total if absolute_total else 0.0),
        )
        for position, (token_id, value) in enumerate(zip(ids, raw, strict=True))
    )
    margin_value = float(margin.detach().to(dtype=torch.float64, device="cpu").item())
    attribution_sum = float(raw.sum().item())
    return MarginAttribution(
        method="gradient_x_input_candidate_margin",
        reference="zero_embedding_local_linearization",
        positive_candidate=positive_candidate,
        negative_candidate=negative_candidate,
        margin=margin_value,
        attribution_sum=attribution_sum,
        absolute_attribution_sum=absolute_total,
        completeness_residual=margin_value - attribution_sum,
        tokens=rows,
    )
