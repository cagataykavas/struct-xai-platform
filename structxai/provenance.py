from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Protocol


class ExperimentSpecLike(Protocol):
    name: str
    prompt: str
    candidates: tuple[str, ...]
    model_name: str


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


def _spec_payload(spec: ExperimentSpecLike) -> dict[str, object]:
    return {
        "name": spec.name,
        "prompt": spec.prompt,
        "candidates": list(spec.candidates),
        "model_name": spec.model_name,
    }


def experiment_fingerprint(spec: ExperimentSpecLike) -> str:
    return _sha256_json(_spec_payload(spec))


def build_provenance(
    spec: ExperimentSpecLike,
    base: dict,
    *,
    requested_device: str | None,
) -> dict[str, Any]:
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


def _as_finite_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _validate_decision_trace(
    analysis: dict[str, Any],
    candidate_labels: tuple[str, ...],
    *,
    context: str,
) -> list[str]:
    """Check that stored layer decisions agree with their candidate scores."""
    errors: list[str] = []
    expected_metric = "first_token_candidate_logit_margin"
    if analysis.get("metric") != expected_metric:
        errors.append(f"{context}_metric_mismatch")

    encoded_candidates = analysis.get("candidates")
    if not isinstance(encoded_candidates, list):
        errors.append(f"{context}_candidates_missing")
    else:
        encoded_labels = [
            row.get("label") if isinstance(row, dict) else None
            for row in encoded_candidates
        ]
        if encoded_labels != list(candidate_labels):
            errors.append(f"{context}_candidate_labels_mismatch")

    layers = analysis.get("layers")
    if not isinstance(layers, list) or not layers:
        errors.append(f"{context}_layers_missing")
        return errors
    if len(candidate_labels) < 2 or len(set(candidate_labels)) != len(candidate_labels):
        errors.append(f"{context}_candidate_labels_invalid")
        return errors

    layer_indices: list[int] = []
    for position, row in enumerate(layers):
        layer_context = f"{context}_layer_{position}"
        if not isinstance(row, dict):
            errors.append(f"{layer_context}_invalid")
            continue

        layer_index = row.get("layer")
        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            errors.append(f"{layer_context}_index_invalid")
        else:
            layer_indices.append(layer_index)

        raw_scores = row.get("candidate_scores")
        if not isinstance(raw_scores, dict) or set(raw_scores) != set(candidate_labels):
            errors.append(f"{layer_context}_candidate_scores_mismatch")
            continue

        scores = {label: _as_finite_float(raw_scores[label]) for label in candidate_labels}
        invalid_labels = [label for label, score in scores.items() if score is None]
        if invalid_labels:
            errors.append(f"{layer_context}_candidate_scores_non_finite")
            continue

        finite_scores = {label: float(scores[label]) for label in candidate_labels}
        ranking = sorted(candidate_labels, key=finite_scores.__getitem__, reverse=True)
        winner, runner_up = ranking[:2]
        expected_values = {
            "winner_score": finite_scores[winner],
            "runner_up_score": finite_scores[runner_up],
            "margin": finite_scores[winner] - finite_scores[runner_up],
        }
        if row.get("winner") != winner:
            errors.append(f"{layer_context}_winner_mismatch")
        if row.get("runner_up") != runner_up:
            errors.append(f"{layer_context}_runner_up_mismatch")
        for field, expected in expected_values.items():
            actual = _as_finite_float(row.get(field))
            if actual is None or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
                errors.append(f"{layer_context}_{field}_mismatch")

    if layer_indices != list(range(len(layers))):
        errors.append(f"{context}_layer_indices_noncontiguous")
    return errors


def validate_artifact(payload: dict) -> dict[str, object]:
    """Validate configuration identity and semantic consistency of an artifact."""
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
    if not isinstance(variants, dict):
        errors.append("invalid_variants")
        variants = {}

    try:
        name = str(experiment["name"])
        prompt = str(experiment["prompt"])
        candidates = tuple(str(item) for item in experiment["candidates"])
        model_name = str(experiment["model_name"])
    except (KeyError, TypeError) as exc:
        errors.append(f"invalid_experiment:{type(exc).__name__}")
        return {"valid": False, "errors": errors}

    expected_fingerprint = _sha256_json(
        {
            "name": name,
            "prompt": prompt,
            "candidates": list(candidates),
            "model_name": model_name,
        }
    )
    if provenance.get("experiment_fingerprint") != expected_fingerprint:
        errors.append("experiment_fingerprint_mismatch")
    if provenance.get("prompt_sha256") != hashlib.sha256(prompt.encode("utf-8")).hexdigest():
        errors.append("prompt_hash_mismatch")
    if provenance.get("candidates_sha256") != _sha256_json(list(candidates)):
        errors.append("candidate_hash_mismatch")
    if base.get("prompt") != prompt:
        errors.append("base_prompt_mismatch")
    if base.get("model") != model_name:
        errors.append("base_model_mismatch")
    if provenance.get("analysis_metric") != base.get("metric"):
        errors.append("provenance_metric_mismatch")

    base_layers = base.get("layers")
    if not isinstance(base_layers, list) or not base_layers:
        errors.append("base_layers_missing")
        return {"valid": not errors, "errors": errors}
    errors.extend(_validate_decision_trace(base, candidates, context="base"))

    base_indices = [row.get("layer") for row in base_layers if isinstance(row, dict)]
    for variant_name, variant_payload in variants.items():
        analysis = variant_payload.get("analysis") if isinstance(variant_payload, dict) else None
        if not isinstance(analysis, dict):
            errors.append(f"variant_analysis_missing:{variant_name}")
            continue
        if analysis.get("model") != model_name:
            errors.append(f"variant_model_mismatch:{variant_name}")
        errors.extend(
            _validate_decision_trace(
                analysis,
                candidates,
                context=f"variant:{variant_name}",
            )
        )
        variant_layers = analysis.get("layers")
        if not isinstance(variant_layers, list):
            errors.append(f"variant_layers_missing:{variant_name}")
            continue
        variant_indices = [row.get("layer") for row in variant_layers if isinstance(row, dict)]
        if variant_indices != base_indices:
            errors.append(f"variant_layer_alignment_mismatch:{variant_name}")

    return {
        "valid": not errors,
        "errors": errors,
        "experiment_fingerprint": provenance.get("experiment_fingerprint"),
        "layers": len(base_layers),
        "variants": len(variants),
    }
