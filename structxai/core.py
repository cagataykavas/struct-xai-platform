from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import torch


@dataclass(frozen=True)
class Candidate:
    label: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class LayerDecision:
    layer: int
    winner: str
    winner_score: float
    runner_up: str
    runner_up_score: float
    margin: float
    candidate_scores: dict[str, float]


@dataclass(frozen=True)
class SignFlip:
    layer: int
    previous_margin: float
    current_margin: float


def candidate_first_token_scores(logits: torch.Tensor, candidates: Sequence[Candidate]) -> dict[str, float]:
    """Score candidates using their first answer token at one projected layer.

    Layer-wise logit-lens analysis cannot faithfully score an arbitrary
    multi-token completion from one hidden state. The platform therefore labels
    this metric explicitly as a *first-token candidate score* instead of
    pretending it is the full sequence probability.
    """
    scores: dict[str, float] = {}
    for candidate in candidates:
        if not candidate.token_ids:
            raise ValueError(f"candidate {candidate.label!r} has no tokens")
        scores[candidate.label] = float(logits[candidate.token_ids[0]].item())
    return scores


def decision_from_scores(layer: int, scores: dict[str, float]) -> LayerDecision:
    if len(scores) < 2:
        raise ValueError("at least two candidates are required")
    ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    (winner, winner_score), (runner_up, runner_up_score) = ranking[:2]
    return LayerDecision(
        layer=layer,
        winner=winner,
        winner_score=float(winner_score),
        runner_up=runner_up,
        runner_up_score=float(runner_up_score),
        margin=float(winner_score - runner_up_score),
        candidate_scores={key: float(value) for key, value in scores.items()},
    )


def layerwise_decisions(
    projected_logits: Iterable[torch.Tensor],
    candidates: Sequence[Candidate],
) -> list[LayerDecision]:
    return [
        decision_from_scores(layer, candidate_first_token_scores(logits, candidates))
        for layer, logits in enumerate(projected_logits)
    ]


def signed_margin(decision: LayerDecision, positive_label: str, negative_label: str) -> float:
    return decision.candidate_scores[positive_label] - decision.candidate_scores[negative_label]


def find_sign_flips(
    decisions: Sequence[LayerDecision],
    positive_label: str,
    negative_label: str,
) -> list[SignFlip]:
    flips: list[SignFlip] = []
    if len(decisions) < 2:
        return flips
    previous = signed_margin(decisions[0], positive_label, negative_label)
    for decision in decisions[1:]:
        current = signed_margin(decision, positive_label, negative_label)
        if previous != 0 and current != 0 and (previous > 0) != (current > 0):
            flips.append(SignFlip(decision.layer, previous, current))
        previous = current
    return flips


def serialize_decisions(decisions: Sequence[LayerDecision]) -> list[dict]:
    return [asdict(item) for item in decisions]
