"""Compact layer-wise interpretability demo for instruction-tuned causal LMs.

This public example inspects the top next-token prediction at every hidden layer,
then compares the trajectory after simple input ablations. It is intentionally
small enough to audit and extend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LayerPrediction:
    layer: int
    token: str
    logit: float


def get_layer_predictions(prompt: str, model, tokenizer, device: str) -> list[LayerPrediction]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    final_norm = model.model.norm
    lm_head = model.lm_head
    predictions: list[LayerPrediction] = []

    for layer_idx, hidden_state in enumerate(outputs.hidden_states):
        final_token = hidden_state[0, -1, :]
        normalized = final_norm(final_token)
        logits = lm_head(normalized)
        token_id = int(torch.argmax(logits).item())
        predictions.append(
            LayerPrediction(
                layer=layer_idx,
                token=tokenizer.decode([token_id]),
                logit=float(logits[token_id].item()),
            )
        )
    return predictions


def compare_ablations(
    base_prompt: str,
    ablations: dict[str, str],
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    device: str | None = None,
) -> dict[str, list[LayerPrediction]]:
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if resolved_device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(resolved_device)
    model.eval()

    result = {"base": get_layer_predictions(base_prompt, model, tokenizer, resolved_device)}
    for label, prompt in ablations.items():
        result[label] = get_layer_predictions(prompt, model, tokenizer, resolved_device)
    return result


def changed_layers(result: dict[str, list[LayerPrediction]], label: str) -> list[int]:
    base = result["base"]
    variant = result[label]
    return [a.layer for a, b in zip(base, variant) if a.token != b.token]


def main() -> None:
    base = "Question: What is the capital of Turkey?\nAnswer:"
    variants = {
        "remove_country": "Question: What is the capital?\nAnswer:",
        "replace_country": "Question: What is the capital of Greece?\nAnswer:",
    }
    result = compare_ablations(base, variants)

    for label in variants:
        print(f"{label}: changed layers = {changed_layers(result, label)}")


if __name__ == "__main__":
    main()
