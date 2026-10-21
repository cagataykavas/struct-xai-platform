from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from structxai.evaluation import evaluate_interventions
from structxai.hf_runner import run_layerwise
from structxai.interventions import Intervention
from structxai.perturbation_integrity import (
    AuditPolicy,
    EvidenceError,
    ExecutionIdentity,
    PerturbationArtifact,
    audit_artifact,
)
from structxai.provenance import build_provenance


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    prompt: str
    candidates: tuple[str, ...]
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass(frozen=True)
class PerturbationAuditSpec:
    """Immutable execution lock and protected prompt span for token audits."""

    model_revision: str
    tokenizer_revision: str
    policy_id: str
    protected_literal: str
    policy: AuditPolicy = field(default_factory=AuditPolicy)

    def __post_init__(self) -> None:
        if not self.protected_literal:
            raise ValueError("protected_literal must be non-empty")
        if not isinstance(self.policy, AuditPolicy):
            raise TypeError("policy must be an AuditPolicy")


class PerturbationAuditRejected(RuntimeError):
    """Raised when a live intervention violates its token-level contract."""


def _exact_token_span(trace: dict[str, Any], start: int, end: int, field: str) -> tuple[int, int]:
    offsets = trace.get("offset_mapping")
    if not isinstance(offsets, list) or start >= end:
        raise EvidenceError(f"{field} must map to a non-empty tokenizer span")
    indices = [
        index
        for index, pair in enumerate(offsets)
        if isinstance(pair, (list, tuple))
        and len(pair) == 2
        and pair[1] > pair[0]
        and pair[0] < end
        and pair[1] > start
    ]
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        raise EvidenceError(f"{field} does not map to contiguous tokens")
    first, last = indices[0], indices[-1]
    if offsets[first][0] != start:
        raise EvidenceError(f"{field} start is not token-aligned")
    if offsets[last][1] != end:
        raise EvidenceError(f"{field} end is not token-aligned")
    return first, last + 1


def _token_sequence(trace: dict[str, Any]) -> dict[str, list[int]]:
    return {
        "input_ids": list(trace["input_ids"]),
        "attention_mask": list(trace["attention_mask"]),
    }


def _execution_identity(
    spec: ExperimentSpec,
    audit_spec: PerturbationAuditSpec,
) -> ExecutionIdentity:
    return ExecutionIdentity(
        model_id=spec.model_name,
        model_revision=audit_spec.model_revision,
        tokenizer_id=spec.model_name,
        tokenizer_revision=audit_spec.tokenizer_revision,
        policy_id=audit_spec.policy_id,
        prompt_sha256=hashlib.sha256(spec.prompt.encode("utf-8")).hexdigest(),
    )


def _validate_trace_compatibility(
    baseline: dict[str, Any],
    variant: dict[str, Any],
    variant_name: str,
) -> None:
    fields = ("vocab_size", "pad_token_id", "mask_token_id", "special_token_ids")
    if any(variant.get(field) != baseline.get(field) for field in fields):
        raise EvidenceError(f"tokenizer metadata changed for variant {variant_name!r}")


def _build_perturbation_artifact(
    spec: ExperimentSpec,
    audit_spec: PerturbationAuditSpec,
    base_trace: dict[str, Any],
    variant_traces: Sequence[tuple[Intervention, dict[str, Any], str]],
    generated_at: datetime,
    identity: ExecutionIdentity,
) -> tuple[PerturbationArtifact, ExecutionIdentity]:
    protected_start = spec.prompt.find(audit_spec.protected_literal)
    if protected_start < 0:
        raise EvidenceError("protected_literal is absent from the baseline prompt")
    if spec.prompt.find(audit_spec.protected_literal, protected_start + 1) >= 0:
        raise EvidenceError("protected_literal must occur exactly once")
    protected_end = protected_start + len(audit_spec.protected_literal)
    protected_span = _exact_token_span(base_trace, protected_start, protected_end, "protected_literal")

    variants: list[dict[str, Any]] = []
    for intervention, trace, changed_prompt in variant_traces:
        _validate_trace_compatibility(base_trace, trace, intervention.name)
        start, end = _exact_token_span(
            base_trace,
            intervention.start,
            intervention.end,
            f"intervention {intervention.name!r}",
        )
        replacement_ids: list[int] = []
        if intervention.kind == "replace":
            changed_end = intervention.start + len(intervention.replacement)
            replacement_start, replacement_end = _exact_token_span(
                trace,
                intervention.start,
                changed_end,
                f"replacement {intervention.name!r}",
            )
            replacement_ids = list(trace["input_ids"][replacement_start:replacement_end])
        if changed_prompt != intervention.apply(spec.prompt):
            raise EvidenceError("variant prompt does not match its intervention")
        variants.append(
            {
                "variant_id": intervention.name,
                "kind": intervention.kind,
                "start": start,
                "end": end,
                "replacement_ids": replacement_ids,
                "sequence": _token_sequence(trace),
            }
        )

    values = {
        "schema_version": "struct-xai-perturbation-artifact/v1",
        "generated_at": generated_at.isoformat(),
        "experiment_id": spec.name,
        **identity.canonical_dict(),
        "vocab_size": base_trace["vocab_size"],
        "pad_token_id": base_trace["pad_token_id"],
        "mask_token_id": base_trace["mask_token_id"],
        "special_token_ids": base_trace["special_token_ids"],
        "baseline": _token_sequence(base_trace),
        "candidate_span": list(protected_span),
        "variants": variants,
    }
    return PerturbationArtifact.from_dict(values), identity


def compare_interventions(
    spec: ExperimentSpec,
    interventions: Sequence[Intervention],
    output_dir: str | Path = "artifacts",
    device: str | None = None,
    *,
    perturbation_audit: PerturbationAuditSpec | None = None,
) -> dict:
    runner_kwargs: dict[str, Any] = {}
    execution_identity = None
    if perturbation_audit is not None:
        if not interventions:
            raise EvidenceError("perturbation audit requires at least one intervention")
        execution_identity = _execution_identity(spec, perturbation_audit)
        runner_kwargs = {
            "model_revision": perturbation_audit.model_revision,
            "tokenizer_revision": perturbation_audit.tokenizer_revision,
            "capture_token_trace": True,
        }
    base = run_layerwise(spec.prompt, spec.candidates, spec.model_name, device=device, **runner_kwargs)
    base_trace = base.pop("_token_trace", None)
    variants: dict[str, dict] = {}
    variant_traces: list[tuple[Intervention, dict[str, Any], str]] = []
    for intervention in interventions:
        changed_prompt = intervention.apply(spec.prompt)
        analysis = run_layerwise(
            changed_prompt,
            spec.candidates,
            spec.model_name,
            device=device,
            **runner_kwargs,
        )
        trace = analysis.pop("_token_trace", None)
        if perturbation_audit is not None:
            if trace is None:
                raise EvidenceError("runner did not return a token trace")
            variant_traces.append((intervention, trace, changed_prompt))
        variants[intervention.name] = {
            "intervention": asdict(intervention),
            "prompt": changed_prompt,
            "analysis": analysis,
        }

    payload = {
        "experiment": asdict(spec),
        "created_at": datetime.now(UTC).isoformat(),
        "provenance": build_provenance(spec, base, requested_device=device),
        "base": base,
        "variants": variants,
        "evaluation": evaluate_interventions(base, variants, spec.candidates),
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if perturbation_audit is not None:
        if base_trace is None:
            raise EvidenceError("runner did not return a baseline token trace")
        artifact, identity = _build_perturbation_artifact(
            spec,
            perturbation_audit,
            base_trace,
            variant_traces,
            datetime.now(UTC),
            execution_identity,
        )
        report = audit_artifact(
            artifact,
            perturbation_audit.policy,
            expected_identity=identity,
        )
        artifact_path = out / f"{spec.name}.perturbations.json"
        report_path = out / f"{spec.name}.perturbation-audit.json"
        artifact_path.write_text(
            json.dumps(artifact.canonical_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if report["status"] != "accepted":
            raise PerturbationAuditRejected(
                f"token perturbation audit rejected the run; inspect {report_path}"
            )
        payload["perturbation_integrity"] = {
            "status": report["status"],
            "artifact_path": str(artifact_path),
            "report_path": str(report_path),
            "artifact_sha256": report["artifact_sha256"],
            "evidence_sha256": report["evidence_sha256"],
            "execution_lock_sha256": report["execution_lock_sha256"],
            "variants": report["metrics"]["variants"],
        }
    path = out / f"{spec.name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["artifact_path"] = str(path)
    return payload


def layer_margin_delta(base: dict, variant: dict, positive: str, negative: str) -> list[dict]:
    rows: list[dict] = []
    for base_layer, variant_layer in zip(base["layers"], variant["layers"], strict=True):
        base_margin = base_layer["candidate_scores"][positive] - base_layer["candidate_scores"][negative]
        variant_margin = (
            variant_layer["candidate_scores"][positive] - variant_layer["candidate_scores"][negative]
        )
        rows.append(
            {
                "layer": base_layer["layer"],
                "base_margin": base_margin,
                "variant_margin": variant_margin,
                "delta": variant_margin - base_margin,
            }
        )
    return rows
