from __future__ import annotations

import pytest
import torch

from structxai.attribution import gradient_input_margin_attribution
from structxai.cli import build_parser


def test_gradient_input_attributes_signed_linear_margin() -> None:
    embeddings = torch.tensor(
        [[1.0, 2.0], [-1.0, 3.0]],
        requires_grad=True,
    )
    weights = torch.tensor([[2.0, -1.0], [0.5, 1.0]])
    positive = (embeddings * weights).sum() + 0.5
    negative = embeddings.sum() * 0.0

    result = gradient_input_margin_attribution(
        embeddings,
        positive,
        negative,
        [101, 202],
        positive_candidate="yes",
        negative_candidate="no",
    )

    assert result.method == "gradient_x_input_candidate_margin"
    assert result.margin == pytest.approx(3.0)
    assert [row.attribution for row in result.tokens] == pytest.approx([0.0, 2.5])
    assert [row.absolute_share for row in result.tokens] == pytest.approx([0.0, 1.0])
    assert result.attribution_sum == pytest.approx(2.5)
    assert result.completeness_residual == pytest.approx(0.5)


def test_zero_attribution_has_finite_zero_shares() -> None:
    embeddings = torch.ones((3, 2), requires_grad=True)
    connected_zero = embeddings.sum() * 0.0

    result = gradient_input_margin_attribution(
        embeddings,
        connected_zero,
        connected_zero,
        [1, 2, 3],
        positive_candidate="A",
        negative_candidate="B",
    )

    assert result.absolute_attribution_sum == 0.0
    assert [row.absolute_share for row in result.tokens] == [0.0, 0.0, 0.0]


def test_attribution_rejects_disconnected_or_malformed_evidence() -> None:
    embeddings = torch.ones((2, 3), requires_grad=True)
    disconnected = torch.tensor(1.0, requires_grad=True)

    with pytest.raises(ValueError, match="disconnected"):
        gradient_input_margin_attribution(
            embeddings,
            disconnected,
            disconnected * 0.0,
            [1, 2],
            positive_candidate="A",
            negative_candidate="B",
        )

    with pytest.raises(ValueError, match="one ID per token"):
        gradient_input_margin_attribution(
            embeddings,
            embeddings.sum(),
            embeddings.sum() * 0.0,
            [1],
            positive_candidate="A",
            negative_candidate="B",
        )

    non_finite = torch.tensor([[float("nan")]], requires_grad=True)
    with pytest.raises(ValueError, match="must be finite"):
        gradient_input_margin_attribution(
            non_finite,
            non_finite.sum(),
            non_finite.sum() * 0.0,
            [1],
            positive_candidate="A",
            negative_candidate="B",
        )


def test_attribute_cli_requires_exactly_two_candidates() -> None:
    parser = build_parser()
    args = parser.parse_args(["attribute", "--prompt", "Question?", "--candidates", " yes", " no"])

    assert args.command == "attribute"
    assert args.candidates == [" yes", " no"]
