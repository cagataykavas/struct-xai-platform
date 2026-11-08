"""Offline architecture-integration tests; randomly initialized models, not NLP quality."""
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, Qwen2Config, Qwen2ForCausalLM

from structxai.hf_runner import project_hidden_states, resolve_projection


def tiny_model(family, seed):
    torch.manual_seed(seed)
    if family == "qwen2":
        model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=31, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        ))
    else:
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=31, n_embd=16, n_layer=2, n_head=2, n_positions=32,
        ))
    model.eval()
    norm, _ = resolve_projection(model)
    # Nonuniform learned affine weights expose the double-normalization error.
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 2.0, 16))
    return model


@pytest.mark.parametrize("family", ["qwen2", "gpt2"])
@pytest.mark.parametrize("seed", [7, 19, 42])
def test_final_projection_matches_native_logits(family, seed):
    model = tiny_model(family, seed)
    with torch.no_grad():
        output = model(torch.tensor([[1, 2, 3, 4]]), output_hidden_states=True, use_cache=False)
    projected = project_hidden_states(model, output.hidden_states)
    assert len(projected) == 3
    torch.testing.assert_close(projected[-1], output.logits[0, -1].float(), atol=0, rtol=0)
    norm, head = resolve_projection(model)
    for index in range(len(projected) - 1):
        torch.testing.assert_close(projected[index], head(norm(output.hidden_states[index][0, -1])).float())
    # Confirm the fixture would fail with the old implementation.
    doubled = head(norm(output.hidden_states[-1][0, -1])).float()
    assert torch.max(torch.abs(doubled - output.logits[0, -1])).item() > 1e-3


def test_empty_hidden_state_sequence_is_rejected():
    with pytest.raises(ValueError, match="at least the final state"):
        project_hidden_states(tiny_model("qwen2", 7), [])
