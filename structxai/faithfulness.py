"""Faithfulness evaluation for progressive attribution-guided deletions."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import pairwise


@dataclass(frozen=True)
class DeletionStep:
    """Candidate-margin evidence after removing a ranked attribution fraction."""

    removed_fraction: float
    candidate_margin: float


@dataclass(frozen=True)
class DeletionFaithfulnessReport:
    """JSON-ready summary and release decision for one deletion curve."""

    steps: int
    baseline_margin: float
    final_margin: float
    final_margin_drop: float
    peak_margin_drop: float
    aopc: float
    monotonicity: float
    first_winner_flip_fraction: float | None
    minimum_aopc: float
    minimum_monotonicity: float
    passed: bool
    reasons: tuple[str, ...]
    curve: tuple[DeletionStep, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_deletion_curve(
    steps: Iterable[DeletionStep],
    *,
    minimum_aopc: float = 0.0,
    minimum_monotonicity: float = 0.8,
    recovery_tolerance: float = 0.0,
) -> DeletionFaithfulnessReport:
    """Evaluate whether ranked deletions progressively suppress a chosen margin.

    The first step must represent the untouched input at removed_fraction=0.
    A positive margin means the explained candidate is preferred to its foil.
    AOPC is trapezoidal area under the margin-drop curve divided by the
    observed deletion range, which supports non-uniform deletion schedules.
    """

    curve = tuple(steps)
    if len(curve) < 3:
        raise ValueError("deletion curve requires at least three steps")
    if not math.isfinite(minimum_aopc):
        raise ValueError("minimum_aopc must be finite")
    if not 0.0 <= minimum_monotonicity <= 1.0:
        raise ValueError("minimum_monotonicity must be between zero and one")
    if not math.isfinite(recovery_tolerance) or recovery_tolerance < 0:
        raise ValueError("recovery_tolerance must be finite and non-negative")

    previous_fraction = -1.0
    for index, step in enumerate(curve):
        if not isinstance(step, DeletionStep):
            raise TypeError(f"steps[{index}] must be a DeletionStep")
        if not math.isfinite(step.removed_fraction) or not 0.0 <= step.removed_fraction <= 1.0:
            raise ValueError(f"steps[{index}].removed_fraction must be finite and within [0, 1]")
        if not math.isfinite(step.candidate_margin):
            raise ValueError(f"steps[{index}].candidate_margin must be finite")
        if step.removed_fraction <= previous_fraction:
            raise ValueError("removed fractions must be strictly increasing")
        previous_fraction = step.removed_fraction

    if curve[0].removed_fraction != 0.0:
        raise ValueError("deletion curve must start at removed_fraction=0")
    if curve[0].candidate_margin <= 0:
        raise ValueError("baseline candidate margin must be positive")
    if curve[-1].removed_fraction <= 0:
        raise ValueError("deletion curve must cover a positive removal fraction")

    baseline_margin = curve[0].candidate_margin
    drops = [baseline_margin - step.candidate_margin for step in curve]
    area = 0.0
    for left, right, left_drop, right_drop in zip(
        curve,
        curve[1:],
        drops,
        drops[1:],
        strict=True,
    ):
        width = right.removed_fraction - left.removed_fraction
        area += width * (left_drop + right_drop) / 2.0
    aopc = area / curve[-1].removed_fraction

    monotonic_transitions = sum(
        right.candidate_margin <= left.candidate_margin + recovery_tolerance
        for left, right in pairwise(curve)
    )
    monotonicity = monotonic_transitions / (len(curve) - 1)
    flip_fraction = next(
        (step.removed_fraction for step in curve[1:] if step.candidate_margin <= 0),
        None,
    )

    reasons: list[str] = []
    if aopc < minimum_aopc:
        reasons.append("insufficient_aopc")
    if monotonicity < minimum_monotonicity:
        reasons.append("non_monotonic_deletion_curve")

    return DeletionFaithfulnessReport(
        steps=len(curve),
        baseline_margin=baseline_margin,
        final_margin=curve[-1].candidate_margin,
        final_margin_drop=drops[-1],
        peak_margin_drop=max(drops),
        aopc=aopc,
        monotonicity=monotonicity,
        first_winner_flip_fraction=flip_fraction,
        minimum_aopc=minimum_aopc,
        minimum_monotonicity=minimum_monotonicity,
        passed=not reasons,
        reasons=tuple(reasons),
        curve=curve,
    )
