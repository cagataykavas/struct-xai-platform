from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from structxai.core import Candidate, candidate_first_token_scores
from structxai.hf_runner import encode_candidates


@dataclass(frozen=True)
class PatchResult:
    layer: int
    baseline_scores: dict[str, float]
    patched_scores: dict[str, float]
    score_deltas: dict[str, float]


def resolve_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise TypeError("unsupported model architecture: cannot resolve transformer layers")


def _final_candidate_scores(logits: torch.Tensor, candidates: Sequence[Candidate]) -> dict[str, float]:
    return candidate_first_token_scores(logits[0, -1, :].float().cpu(), candidates)


def patch_final_position(
    source_prompt: str,
    target_prompt: str,
    layer_index: int,
    candidate_labels: Sequence[str],
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    device: str | None = None,
) -> PatchResult:
    """Patch the target prompt's final-position residual state from a source run.

    This deliberately implements one narrow, inspectable activation-patching
    experiment. It patches a single sequence position at one transformer layer
    and reports how final candidate logits change.
    """
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if resolved_device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(resolved_device)
    model.eval()
    layers = resolve_layers(model)
    if not 0 <= layer_index < len(layers):
        raise IndexError(f"layer_index={layer_index} outside [0, {len(layers) - 1}]")

    candidates = encode_candidates(tokenizer, candidate_labels)
    source_inputs = tokenizer(source_prompt, return_tensors="pt").to(resolved_device)
    target_inputs = tokenizer(target_prompt, return_tensors="pt").to(resolved_device)

    captured: dict[str, torch.Tensor] = {}

    def capture_hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["activation"] = hidden[:, -1, :].detach().clone()

    handle = layers[layer_index].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            model(**source_inputs, use_cache=False)
    finally:
        handle.remove()

    with torch.no_grad():
        baseline_output = model(**target_inputs, use_cache=False)
    baseline_scores = _final_candidate_scores(baseline_output.logits, candidates)

    def patch_hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0].clone()
            hidden[:, -1, :] = captured["activation"].to(hidden.dtype)
            return (hidden, *output[1:])
        hidden = output.clone()
        hidden[:, -1, :] = captured["activation"].to(hidden.dtype)
        return hidden

    handle = layers[layer_index].register_forward_hook(patch_hook)
    try:
        with torch.no_grad():
            patched_output = model(**target_inputs, use_cache=False)
    finally:
        handle.remove()

    patched_scores = _final_candidate_scores(patched_output.logits, candidates)
    deltas = {label: patched_scores[label] - baseline_scores[label] for label in baseline_scores}
    return PatchResult(layer_index, baseline_scores, patched_scores, deltas)


def serialize_patch(result: PatchResult) -> dict:
    return asdict(result)
