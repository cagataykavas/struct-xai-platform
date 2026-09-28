from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from structxai.core import Candidate
from structxai.hf_runner import run_layerwise
from structxai.sequence_scoring import score_candidate_sequences


class TransitionModel:
    """Tiny causal LM whose next-token logits depend only on the current token."""

    def __init__(self, transitions: torch.Tensor) -> None:
        self.transitions = transitions
        self.config = SimpleNamespace(vocab_size=transitions.shape[1])

    def __call__(self, *, input_ids, attention_mask, use_cache):
        assert attention_mask.shape == input_ids.shape
        assert use_cache is False
        return SimpleNamespace(logits=self.transitions[input_ids])


def _model() -> TransitionModel:
    # After prompt token 0, candidate A's first token (1) beats B's (3).
    # A's continuation token (2) is then very unlikely, while B's (4) is likely.
    transitions = torch.tensor(
        [
            [-8.0, 5.0, -8.0, 4.0, -8.0],
            [0.0, 0.0, -6.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-8.0, -8.0, -8.0, -8.0, 5.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    return TransitionModel(transitions)


def _candidates() -> list[Candidate]:
    return [Candidate("A", (1, 2)), Candidate("B", (3, 4))]


def test_complete_sequence_scoring_detects_first_token_proxy_reversal() -> None:
    result = score_candidate_sequences(_model(), torch.tensor([[0]]), _candidates())

    assert result["rankings"]["first_token_log_probability"] == ["A", "B"]
    assert result["rankings"]["total_log_probability"] == ["B", "A"]
    assert result["first_token_winner_matches_total_winner"] is False
    rows = {row["label"]: row for row in result["candidates"]}
    assert rows["A"]["token_count"] == 2
    assert rows["A"]["total_log_probability"] < rows["B"]["total_log_probability"]


def test_scores_variable_length_candidates_in_one_padded_batch() -> None:
    class RecordingModel(TransitionModel):
        def __init__(self) -> None:
            super().__init__(_model().transitions)
            self.calls = 0
            self.last_attention_mask = None

        def __call__(self, *, input_ids, attention_mask, use_cache):
            self.calls += 1
            self.last_attention_mask = attention_mask
            return super().__call__(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
            )

    model = RecordingModel()
    candidates = [Candidate("short", (1,)), Candidate("long", (3, 4))]

    result = score_candidate_sequences(model, torch.tensor([[0]]), candidates)

    assert model.calls == 1
    assert model.last_attention_mask.tolist() == [[1, 1, 0], [1, 1, 1]]
    assert [row["token_count"] for row in result["candidates"]] == [1, 2]


def test_reports_total_and_length_normalized_rankings() -> None:
    result = score_candidate_sequences(_model(), torch.tensor([[0]]), _candidates())

    assert set(result["rankings"]) == {
        "total_log_probability",
        "mean_log_probability",
        "first_token_log_probability",
    }
    for row in result["candidates"]:
        assert row["mean_log_probability"] == pytest.approx(row["total_log_probability"] / row["token_count"])
        assert row["minimum_token_log_probability"] <= row["first_token_log_probability"]


def test_ranking_ties_are_deterministic_by_label() -> None:
    transitions = torch.zeros((4, 4))
    candidates = [Candidate("zeta", (1,)), Candidate("alpha", (2,))]

    result = score_candidate_sequences(TransitionModel(transitions), torch.tensor([[0]]), candidates)

    assert result["rankings"]["total_log_probability"] == ["alpha", "zeta"]


@pytest.mark.parametrize(
    ("prompt", "candidates", "error"),
    [
        (torch.empty((1, 0), dtype=torch.long), _candidates(), "cannot be empty"),
        (torch.tensor([[0]]), [Candidate("A", (1,))], "candidate count"),
        (
            torch.tensor([[0]]),
            [Candidate("A", (1,)), Candidate("A", (2,))],
            "must be unique",
        ),
        (
            torch.tensor([[0]]),
            [Candidate("A", (1,)), Candidate("B", ())],
            "has no tokens",
        ),
        (
            torch.tensor([[0]]),
            [Candidate("A", (1,)), Candidate("B", (-1,))],
            "invalid token id",
        ),
    ],
)
def test_rejects_malformed_inputs(prompt, candidates, error) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        score_candidate_sequences(_model(), prompt, candidates)


def test_rejects_candidate_token_budget_overflow() -> None:
    candidates = [Candidate("A", (1, 2)), Candidate("B", (3,))]
    with pytest.raises(ValueError, match="token budget"):
        score_candidate_sequences(
            _model(),
            torch.tensor([[0]]),
            candidates,
            max_candidate_tokens=1,
        )


def test_rejects_padded_or_misaligned_prompt_masks() -> None:
    prompt = torch.tensor([[0, 1]])
    with pytest.raises(ValueError, match="does not accept padded"):
        score_candidate_sequences(
            _model(),
            prompt,
            _candidates(),
            prompt_attention_mask=torch.tensor([[1, 0]]),
        )
    with pytest.raises(ValueError, match="must match"):
        score_candidate_sequences(
            _model(),
            prompt,
            _candidates(),
            prompt_attention_mask=torch.tensor([[1]]),
        )


def test_rejects_out_of_vocabulary_candidate_token() -> None:
    candidates = [Candidate("A", (1,)), Candidate("B", (99,))]
    with pytest.raises(ValueError, match="outside the model vocabulary"):
        score_candidate_sequences(_model(), torch.tensor([[0]]), candidates)


def test_rejects_attempts_to_raise_hard_resource_budgets() -> None:
    with pytest.raises(ValueError, match="max_candidates"):
        score_candidate_sequences(
            _model(),
            torch.tensor([[0]]),
            _candidates(),
            max_candidates=65,
        )
    with pytest.raises(ValueError, match="max_candidate_tokens"):
        score_candidate_sequences(
            _model(),
            torch.tensor([[0]]),
            _candidates(),
            max_candidate_tokens=129,
        )


def test_rejects_unbounded_candidate_label() -> None:
    candidates = [Candidate("A" * 513, (1,)), Candidate("B", (3,))]
    with pytest.raises(ValueError, match="512 UTF-8 bytes"):
        score_candidate_sequences(_model(), torch.tensor([[0]]), candidates)


def test_run_layerwise_attaches_sequence_evidence(monkeypatch) -> None:
    class Batch(dict):
        def to(self, _device):
            return self

    class Tokenizer:
        def __call__(self, _prompt, *, return_tensors):
            assert return_tensors == "pt"
            return Batch(
                input_ids=torch.tensor([[0]]),
                attention_mask=torch.tensor([[1]]),
            )

        def encode(self, label, *, add_special_tokens):
            assert add_special_tokens is False
            return {"A": [1, 2], "B": [3, 4]}[label]

    class LayerwiseModel(TransitionModel):
        def __init__(self) -> None:
            super().__init__(_model().transitions)
            self.model = SimpleNamespace(norm=torch.nn.Identity())
            self.lm_head = torch.nn.Identity()

        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(
            self,
            *,
            input_ids,
            attention_mask,
            use_cache,
            output_hidden_states=False,
        ):
            result = super().__call__(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
            )
            if output_hidden_states:
                hidden = torch.zeros((1, input_ids.shape[1], self.config.vocab_size))
                result.hidden_states = (hidden, hidden)
            return result

    monkeypatch.setattr(
        "structxai.hf_runner.AutoTokenizer.from_pretrained",
        lambda _name: Tokenizer(),
    )
    monkeypatch.setattr(
        "structxai.hf_runner.AutoModelForCausalLM.from_pretrained",
        lambda _name, **_kwargs: LayerwiseModel(),
    )

    result = run_layerwise("prompt", ["A", "B"], model_name="tiny", device="cpu")

    sequence = result["final_candidate_sequence_scoring"]
    assert sequence["rankings"]["first_token_log_probability"] == ["A", "B"]
    assert sequence["rankings"]["total_log_probability"] == ["B", "A"]
