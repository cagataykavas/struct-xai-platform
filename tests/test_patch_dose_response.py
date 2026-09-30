from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from structxai.core import Candidate
from structxai.patch_dose_response import (
    PatchSweepError,
    PatchSweepSpec,
    SweepPolicy,
    run_residual_patch_sweep,
)


class TinyBlock(torch.nn.Module):
    def __init__(self, width: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(torch.randn(width, width, generator=generator) * 0.1)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor]:
        return (hidden + torch.tanh(hidden @ self.weight),)


class TinyCausalLM(torch.nn.Module):
    def __init__(self, *, vocab: int = 11, width: int = 6, depth: int = 3) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.embedding = torch.nn.Embedding(vocab, width)
        self.model = SimpleNamespace(
            layers=torch.nn.ModuleList([TinyBlock(width, index + 20) for index in range(depth)])
        )
        self.layers = self.model.layers
        self.head = torch.nn.Linear(width, vocab, bias=False)

    def forward(self, *, input_ids: torch.Tensor, use_cache: bool) -> SimpleNamespace:
        assert use_cache is False
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return SimpleNamespace(logits=self.head(hidden))


class ExplodingTinyLM(TinyCausalLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, *, input_ids: torch.Tensor, use_cache: bool) -> SimpleNamespace:
        self.calls += 1
        if self.calls == 3:
            raise RuntimeError("simulated remote model failure")
        return super().forward(input_ids=input_ids, use_cache=use_cache)


class CallDriftTinyLM(TinyCausalLM):
    def __init__(self, drift_call: int) -> None:
        super().__init__()
        self.calls = 0
        self.drift_call = drift_call

    def forward(self, *, input_ids: torch.Tensor, use_cache: bool) -> SimpleNamespace:
        self.calls += 1
        output = super().forward(input_ids=input_ids, use_cache=use_cache)
        if self.calls == self.drift_call:
            output.logits = output.logits.clone()
            output.logits[0, -1, 1] += 1.0
        return output


class NonFiniteTinyLM(TinyCausalLM):
    def forward(self, *, input_ids: torch.Tensor, use_cache: bool) -> SimpleNamespace:
        output = super().forward(input_ids=input_ids, use_cache=use_cache)
        output.logits = output.logits.clone()
        output.logits[0, -1, 1] = float("nan")
        return output


def spec(**overrides: object) -> PatchSweepSpec:
    values = {
        "model_revision": "sha256:tiny-model-7",
        "layer_indices": (0, 2),
        "alphas": (0.0, 0.5, 1.0),
        "source_position": -1,
        "target_position": -1,
        "score_position": -1,
        "positive_candidate": Candidate("positive", (1,)),
        "negative_candidate": Candidate("negative", (2,)),
        **overrides,
    }
    return PatchSweepSpec(**values)


def inputs() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor([[3, 4, 5]]), torch.tensor([[3, 4, 6]])


def run(**kwargs: object):
    source, target = inputs()
    return run_residual_patch_sweep(
        TinyCausalLM(),
        source_input_ids=source,
        target_input_ids=target,
        spec=spec(),
        **kwargs,
    )


def test_runs_bounded_layer_and_alpha_sweep() -> None:
    result = run()

    assert len(result.points) == 6
    assert [(point.layer, point.alpha) for point in result.points] == [
        (0, 0.0),
        (0, 0.5),
        (0, 1.0),
        (2, 0.0),
        (2, 0.5),
        (2, 1.0),
    ]
    assert result.model_calls == 9
    assert result.target_margin == pytest.approx(result.repeated_target_margin)
    assert len(result.evidence_sha256) == 64


def test_zero_dose_reconciles_with_target_baseline() -> None:
    result = run()

    for point in result.points:
        if point.alpha == 0.0:
            assert point.margin == pytest.approx(result.target_margin)
            assert point.margin_delta == pytest.approx(0.0)
            assert point.normalized_recovery == pytest.approx(0.0)


def test_sweep_is_deterministic_and_input_bound() -> None:
    first = run()
    second = run()

    assert first.as_dict() == second.as_dict()
    assert first.source_input_sha256 != first.target_input_sha256
    assert len(first.source_input_sha256) == 64
    assert len(first.sweep_spec_sha256) == 64


def test_spec_digest_binds_candidate_tokens_and_positions() -> None:
    source, target = inputs()
    base = run()
    changed_candidate = run_residual_patch_sweep(
        TinyCausalLM(),
        source_input_ids=source,
        target_input_ids=target,
        spec=spec(positive_candidate=Candidate("positive", (7,))),
    )
    changed_position = run_residual_patch_sweep(
        TinyCausalLM(),
        source_input_ids=source,
        target_input_ids=target,
        spec=spec(source_position=0),
    )

    assert base.sweep_spec_sha256 != changed_candidate.sweep_spec_sha256
    assert base.sweep_spec_sha256 != changed_position.sweep_spec_sha256


def test_training_state_and_parameters_are_preserved() -> None:
    model = TinyCausalLM()
    model.train()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    source, target = inputs()

    run_residual_patch_sweep(
        model,
        source_input_ids=source,
        target_input_ids=target,
        spec=spec(),
    )

    assert model.training is True
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_hooks_are_removed_after_success() -> None:
    model = TinyCausalLM()
    source, target = inputs()

    run_residual_patch_sweep(
        model,
        source_input_ids=source,
        target_input_ids=target,
        spec=spec(),
    )

    assert all(not layer._forward_hooks for layer in model.layers)


def test_hooks_are_removed_after_model_failure() -> None:
    model = ExplodingTinyLM()
    source, target = inputs()

    with pytest.raises(RuntimeError, match="simulated remote model failure"):
        run_residual_patch_sweep(
            model,
            source_input_ids=source,
            target_input_ids=target,
            spec=spec(),
        )

    assert all(not layer._forward_hooks for layer in model.layers)
    assert model.training is True


def test_zero_dose_control_detects_forward_drift() -> None:
    model = CallDriftTinyLM(drift_call=3)
    source, target = inputs()

    with pytest.raises(PatchSweepError, match="ZERO_DOSE_BASELINE_MISMATCH"):
        run_residual_patch_sweep(
            model,
            source_input_ids=source,
            target_input_ids=target,
            spec=spec(),
        )


def test_final_baseline_repeat_detects_late_drift() -> None:
    model = CallDriftTinyLM(drift_call=9)
    source, target = inputs()

    with pytest.raises(PatchSweepError, match="TARGET_BASELINE_DRIFT"):
        run_residual_patch_sweep(
            model,
            source_input_ids=source,
            target_input_ids=target,
            spec=spec(),
        )


def test_non_finite_model_output_is_rejected() -> None:
    source, target = inputs()

    with pytest.raises(PatchSweepError, match="NON_FINITE_MODEL_OUTPUT"):
        run_residual_patch_sweep(
            NonFiniteTinyLM(),
            source_input_ids=source,
            target_input_ids=target,
            spec=spec(),
        )


def test_hidden_width_budget_is_enforced_during_capture() -> None:
    model = TinyCausalLM(width=6)
    source, target = inputs()

    with pytest.raises(PatchSweepError, match="HIDDEN_WIDTH_BUDGET_EXCEEDED"):
        run_residual_patch_sweep(
            model,
            source_input_ids=source,
            target_input_ids=target,
            spec=spec(),
            policy=SweepPolicy(max_hidden_width=5),
        )
    assert all(not layer._forward_hooks for layer in model.layers)


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"model_revision": "latest"}, "INVALID_MODEL_REVISION"),
        ({"layer_indices": (2, 0)}, "LAYER_INDICES_NOT_STRICTLY_INCREASING"),
        ({"layer_indices": (0, 0)}, "LAYER_INDICES_NOT_STRICTLY_INCREASING"),
        ({"alphas": (0.0, 1.0)}, "INVALID_ALPHA_LADDER"),
        ({"alphas": (0.1, 0.5, 1.0)}, "ALPHA_ENDPOINTS_REQUIRED"),
        ({"alphas": (0.0, 0.8, 0.5, 1.0)}, "ALPHAS_NOT_STRICTLY_INCREASING"),
        (
            {"negative_candidate": Candidate("positive", (2,))},
            "DUPLICATE_CANDIDATE_LABEL",
        ),
        (
            {"negative_candidate": Candidate("negative", (1,))},
            "DUPLICATE_CANDIDATE_TOKEN",
        ),
    ],
)
def test_invalid_sweep_specs_fail_closed(overrides: dict[str, object], code: str) -> None:
    with pytest.raises(PatchSweepError, match=code):
        spec(**overrides)


@pytest.mark.parametrize(
    ("source", "target", "code"),
    [
        (torch.tensor([1, 2]), torch.tensor([[1, 2]]), "INVALID_INPUT_SHAPE"),
        (torch.tensor([[1.0, 2.0]]), torch.tensor([[1, 2]]), "INVALID_INPUT_DTYPE"),
        (torch.tensor([[1, -2]]), torch.tensor([[1, 2]]), "INVALID_INPUT_TOKEN"),
        (torch.tensor([], dtype=torch.long).reshape(1, 0), torch.tensor([[1]]), "SEQUENCE_TOKEN"),
    ],
)
def test_invalid_inputs_fail_closed(source: torch.Tensor, target: torch.Tensor, code: str) -> None:
    with pytest.raises(PatchSweepError, match=code):
        run_residual_patch_sweep(
            TinyCausalLM(),
            source_input_ids=source,
            target_input_ids=target,
            spec=spec(),
        )


def test_positions_and_layers_are_validated() -> None:
    source, target = inputs()
    with pytest.raises(PatchSweepError, match="LAYER_INDEX_OUT_OF_RANGE"):
        run_residual_patch_sweep(
            TinyCausalLM(),
            source_input_ids=source,
            target_input_ids=target,
            spec=spec(layer_indices=(0, 3)),
        )
    with pytest.raises(PatchSweepError, match="TARGET_POSITION_OUT_OF_RANGE"):
        run_residual_patch_sweep(
            TinyCausalLM(),
            source_input_ids=source,
            target_input_ids=target,
            spec=spec(target_position=8),
        )


def test_candidate_vocabulary_range_is_checked() -> None:
    source, target = inputs()
    with pytest.raises(PatchSweepError, match="CANDIDATE_TOKEN_OUT_OF_RANGE"):
        run_residual_patch_sweep(
            TinyCausalLM(),
            source_input_ids=source,
            target_input_ids=target,
            spec=spec(positive_candidate=Candidate("positive", (99,))),
        )


def test_resource_budgets_fail_closed_before_sweep() -> None:
    source, target = inputs()
    with pytest.raises(PatchSweepError, match="MODEL_CALL_BUDGET_EXCEEDED"):
        run_residual_patch_sweep(
            TinyCausalLM(),
            source_input_ids=source,
            target_input_ids=target,
            spec=spec(),
            policy=SweepPolicy(max_model_calls=8),
        )
    with pytest.raises(PatchSweepError, match="LAYER_BUDGET_EXCEEDED"):
        run_residual_patch_sweep(
            TinyCausalLM(),
            source_input_ids=source,
            target_input_ids=target,
            spec=spec(),
            policy=SweepPolicy(max_layers=1),
        )


def test_weak_source_target_contrast_is_rejected() -> None:
    source, _ = inputs()
    with pytest.raises(PatchSweepError, match="WEAK_SOURCE_TARGET_MARGIN_GAP"):
        run_residual_patch_sweep(
            TinyCausalLM(),
            source_input_ids=source,
            target_input_ids=source.clone(),
            spec=spec(),
        )


def test_report_does_not_contain_raw_candidate_labels_or_inputs() -> None:
    source, target = inputs()
    result = run_residual_patch_sweep(
        TinyCausalLM(),
        source_input_ids=source,
        target_input_ids=target,
        spec=spec(
            positive_candidate=Candidate("secret_candidate_a", (1,)),
            negative_candidate=Candidate("secret_candidate_b", (2,)),
        ),
    )
    serialized = str(result.as_dict())

    assert "secret_candidate_a" not in serialized
    assert "secret_candidate_b" not in serialized
    assert "[3, 4, 5]" not in serialized
    assert "tensor" not in serialized
