"""Measure logit-projection parity on a pinned real checkpoint, not task accuracy."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from structxai.hf_runner import project_hidden_states, resolve_projection

PROMPTS = [
    "Türkiye'nin başkenti hangi şehirdir?",
    "İki ile üç toplandığında sonuç kaçtır?",
    "Bir metindeki ana fikri belirlemek ne demektir?",
    "Su deniz seviyesinde yaklaşık kaç derecede kaynar?",
    "Gece gökyüzünde görünen doğal uydu hangisidir?",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.float32,
    ).eval()
    norm, head = resolve_projection(model)
    rows = []
    for prompt in PROMPTS:
        encoded = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model(**encoded, output_hidden_states=True, use_cache=False)
            fixed = project_hidden_states(model, output.hidden_states)[-1]
            legacy = head(norm(output.hidden_states[-1][0, -1])).float().cpu()
            native = output.logits[0, -1].float().cpu()
        # Standard PyTorch float32 tolerance accommodates GEMV/GEMM rounding.
        torch.testing.assert_close(fixed, native, rtol=1.3e-6, atol=1e-5)
        rows.append({
            "prompt": prompt,
            "tokens": int(encoded.input_ids.shape[-1]),
            "hidden_states": len(output.hidden_states),
            "vocabulary_logits": int(native.numel()),
            "legacy_max_absolute_error": float((legacy - native).abs().max()),
            "fixed_max_absolute_error": float((fixed - native).abs().max()),
            "legacy_top1_matches_native": bool(legacy.argmax() == native.argmax()),
            "fixed_top1_matches_native": bool(fixed.argmax() == native.argmax()),
        })
    evidence = {
        "schema": "hf-projection-parity/v1",
        "purpose": "Final projected logits versus the same forward pass native logits; not NLP accuracy",
        "model": args.model, "revision": args.revision,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff"])).hexdigest(),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "transformers": transformers.__version__, "device": "cpu", "dtype": "float32",
                    "threads": 2, "seed": 42},
        "prompt_sha256": hashlib.sha256(json.dumps(PROMPTS, ensure_ascii=False).encode()).hexdigest(),
        "rows": rows,
        "parity_tolerance": {"rtol": 1.3e-6, "atol": 1e-5},
        "maximum_fixed_error": max(r["fixed_max_absolute_error"] for r in rows),
        "maximum_legacy_error": max(r["legacy_max_absolute_error"] for r in rows),
        "limitation": "Five fixed Turkish raw prompts, one checkpoint, CPU float32; no generated-answer quality or causal faithfulness claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(evidence, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
