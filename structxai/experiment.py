from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from structxai.hf_runner import run_layerwise
from structxai.interventions import Intervention


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    prompt: str
    candidates: tuple[str, ...]
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"


def compare_interventions(
    spec: ExperimentSpec,
    interventions: Sequence[Intervention],
    output_dir: str | Path = "artifacts",
    device: str | None = None,
) -> dict:
    base = run_layerwise(spec.prompt, spec.candidates, spec.model_name, device=device)
    variants: dict[str, dict] = {}
    for intervention in interventions:
        changed_prompt = intervention.apply(spec.prompt)
        variants[intervention.name] = {
            "intervention": asdict(intervention),
            "prompt": changed_prompt,
            "analysis": run_layerwise(
                changed_prompt,
                spec.candidates,
                spec.model_name,
                device=device,
            ),
        }

    payload = {
        "experiment": asdict(spec),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base": base,
        "variants": variants,
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{spec.name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["artifact_path"] = str(path)
    return payload


def layer_margin_delta(base: dict, variant: dict, positive: str, negative: str) -> list[dict]:
    rows: list[dict] = []
    for base_layer, variant_layer in zip(base["layers"], variant["layers"]):
        base_margin = base_layer["candidate_scores"][positive] - base_layer["candidate_scores"][negative]
        variant_margin = variant_layer["candidate_scores"][positive] - variant_layer["candidate_scores"][negative]
        rows.append(
            {
                "layer": base_layer["layer"],
                "base_margin": base_margin,
                "variant_margin": variant_margin,
                "delta": variant_margin - base_margin,
            }
        )
    return rows
