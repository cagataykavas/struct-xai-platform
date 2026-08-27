from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from structxai.experiment import ExperimentSpec


def _sha256_json(payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def experiment_fingerprint(spec: ExperimentSpec) -> str:
    return _sha256_json(asdict(spec))


def build_provenance(spec: ExperimentSpec, base: dict, *, requested_device: str | None) -> dict[str, Any]:
    return {
        "schema": "struct-xai-experiment/v1",
        "experiment_fingerprint": experiment_fingerprint(spec),
        "prompt_sha256": hashlib.sha256(spec.prompt.encode("utf-8")).hexdigest(),
        "candidates_sha256": _sha256_json(list(spec.candidates)),
        "model_name": spec.model_name,
        "analysis_metric": base.get("metric"),
        "requested_device": requested_device,
        "resolved_device": base.get("device"),
        "dtype": base.get("dtype"),
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": sys.platform,
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "struct_xai_platform": _package_version("struct-xai-platform"),
        },
        "source_revision": os.getenv("GITHUB_SHA"),
    }


def validate_artifact(payload: dict) -> dict[str, object]:
    """Validate configuration identity and structural consistency of an artifact."""
    errors: list[str] = []
    experiment = payload.get("experiment")
    provenance = payload.get("provenance")
    base = payload.get("base")
    variants = payload.get("variants", {})

    if not isinstance(experiment, dict):
        errors.append("missing_experiment")
        return {"valid": False, "errors": errors}
    if not isinstance(provenance, dict):
        errors.append("missing_provenance")
        return {"valid": False, "errors": errors}
    if not isinstance(base, dict):
        errors.append("missing_base_analysis")
        return {"valid": False, "errors": errors}

    try:
        spec = ExperimentSpec(
            name=str(experiment["name"]),
            prompt=str(experiment["prompt"]),
            candidates=tuple(str(item) for item in experiment["candidates"]),
            model_name=str(experiment["model_name"]),
        )
    except (KeyError, TypeError) as exc:
        errors.append(f"invalid_experiment:{type(exc).__name__}")
        return {"valid": False, "errors": errors}

    if provenance.get("experiment_fingerprint") != experiment_fingerprint(spec):
        errors.append("experiment_fingerprint_mismatch")
    if provenance.get("prompt_sha256") != hashlib.sha256(spec.prompt.encode("utf-8")).hexdigest():
        errors.append("prompt_hash_mismatch")
    if provenance.get("candidates_sha256") != _sha256_json(list(spec.candidates)):
        errors.append("candidate_hash_mismatch")
    if base.get("prompt") != spec.prompt:
        errors.append("base_prompt_mismatch")
    if base.get("model") != spec.model_name:
        errors.append("base_model_mismatch")

    base_layers = base.get("layers")
    if not isinstance(base_layers, list) or not base_layers:
        errors.append("base_layers_missing")
        return {"valid": not errors, "errors": errors}

    base_indices = [row.get("layer") for row in base_layers if isinstance(row, dict)]
    for name, variant_payload in variants.items():
        analysis = variant_payload.get("analysis") if isinstance(variant_payload, dict) else None
        if not isinstance(analysis, dict):
            errors.append(f"variant_analysis_missing:{name}")
            continue
        if analysis.get("model") != spec.model_name:
            errors.append(f"variant_model_mismatch:{name}")
        variant_layers = analysis.get("layers")
        if not isinstance(variant_layers, list):
            errors.append(f"variant_layers_missing:{name}")
            continue
        variant_indices = [row.get("layer") for row in variant_layers if isinstance(row, dict)]
        if variant_indices != base_indices:
            errors.append(f"variant_layer_alignment_mismatch:{name}")

    return {
        "valid": not errors,
        "errors": errors,
        "experiment_fingerprint": provenance.get("experiment_fingerprint"),
        "layers": len(base_layers),
        "variants": len(variants),
    }
