from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from structxai.core import Candidate, find_sign_flips, layerwise_decisions, serialize_decisions


def resolve_projection(model):
    """Resolve the final normalization layer and LM head for common decoder LMs."""
    if hasattr(model, "model") and hasattr(model.model, "norm") and hasattr(model, "lm_head"):
        return model.model.norm, model.lm_head
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f") and hasattr(model, "lm_head"):
        return model.transformer.ln_f, model.lm_head
    raise TypeError("unsupported model architecture: cannot resolve final norm + lm_head")


def encode_candidates(tokenizer, labels: Sequence[str]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for label in labels:
        token_ids = tuple(tokenizer.encode(label, add_special_tokens=False))
        if not token_ids:
            raise ValueError(f"candidate {label!r} tokenized to an empty sequence")
        candidates.append(Candidate(label=label, token_ids=token_ids))
    return candidates


def project_hidden_states(model, hidden_states: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    final_norm, lm_head = resolve_projection(model)
    projected: list[torch.Tensor] = []
    with torch.no_grad():
        for hidden in hidden_states:
            final_token = hidden[0, -1, :]
            projected.append(lm_head(final_norm(final_token)).float().cpu())
    return projected


def run_layerwise(
    prompt: str,
    candidate_labels: Sequence[str],
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    device: str | None = None,
    *,
    model_revision: str | None = None,
    tokenizer_revision: str | None = None,
    capture_token_trace: bool = False,
) -> dict:
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if resolved_device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=tokenizer_revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        torch_dtype=dtype,
    ).to(resolved_device)
    model.eval()

    if capture_token_trace:
        encoded = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
        offsets = encoded.pop("offset_mapping")
        input_ids = encoded["input_ids"][0].tolist()
        attention_mask = encoded["attention_mask"][0].tolist()
        offset_mapping = [tuple(pair) for pair in offsets[0].tolist()]
    else:
        encoded = tokenizer(prompt, return_tensors="pt")
    encoded = encoded.to(resolved_device)
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True, use_cache=False)

    candidates = encode_candidates(tokenizer, candidate_labels)
    projected = project_hidden_states(model, outputs.hidden_states)
    decisions = layerwise_decisions(projected, candidates)

    result = {
        "model": model_name,
        "prompt": prompt,
        "candidates": [asdict(candidate) for candidate in candidates],
        "metric": "first_token_candidate_logit_margin",
        "device": resolved_device,
        "dtype": str(dtype).removeprefix("torch."),
        "layers": serialize_decisions(decisions),
    }
    if len(candidates) == 2:
        result["sign_flips"] = [
            asdict(item) for item in find_sign_flips(decisions, candidates[0].label, candidates[1].label)
        ]
    if capture_token_trace:
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id for perturbation evidence")
        result["_token_trace"] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offset_mapping,
            "vocab_size": len(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
            "mask_token_id": tokenizer.mask_token_id,
            "special_token_ids": sorted(set(tokenizer.all_special_ids)),
        }
    return result
