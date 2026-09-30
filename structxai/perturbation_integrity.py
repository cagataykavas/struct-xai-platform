"""Fail-closed integrity audit for token-level XAI perturbations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ARTIFACT_SCHEMA = "struct-xai-perturbation-artifact/v1"
REPORT_SCHEMA = "struct-xai-perturbation-report/v1"
MAX_INPUT_BYTES = 2 * 1024 * 1024
MAX_REPORTED_FINDINGS = 2048
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")
REVISION = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
KINDS = {"delete", "replace", "mask"}


class EvidenceError(ValueError):
    """Raised when evidence cannot be interpreted safely."""


@dataclass(frozen=True)
class AuditPolicy:
    max_evidence_age_seconds: int = 3600
    max_future_skew_seconds: int = 30
    max_variants: int = 1024
    max_sequence_tokens: int = 4096
    max_total_tokens: int = 1_000_000

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values.values()
        ):
            raise ValueError("policy values must be non-negative integers")
        if self.max_variants < 1 or self.max_sequence_tokens < 1 or self.max_total_tokens < 1:
            raise ValueError("variant and token budgets must be positive")


@dataclass(frozen=True)
class TokenSequence:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]

    def active(self, pad_token_id: int) -> tuple[int, ...]:
        active_length = sum(self.attention_mask)
        if self.attention_mask != (1,) * active_length + (0,) * (len(self.attention_mask) - active_length):
            raise EvidenceError("attention_mask must be contiguous right padding")
        if any(token != pad_token_id for token in self.input_ids[active_length:]):
            raise EvidenceError("masked positions must contain pad_token_id")
        if any(token == pad_token_id for token in self.input_ids[:active_length]):
            raise EvidenceError("active positions cannot contain pad_token_id")
        return self.input_ids[:active_length]

    def canonical_dict(self) -> dict[str, Any]:
        return {"input_ids": list(self.input_ids), "attention_mask": list(self.attention_mask)}


@dataclass(frozen=True)
class PerturbationVariant:
    variant_id: str
    kind: str
    start: int
    end: int
    replacement_ids: tuple[int, ...]
    sequence: TokenSequence

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "replacement_ids": list(self.replacement_ids),
            "sequence": self.sequence.canonical_dict(),
        }


@dataclass(frozen=True)
class PerturbationArtifact:
    generated_at: datetime
    experiment_id: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    policy_id: str
    prompt_sha256: str
    vocab_size: int
    pad_token_id: int
    mask_token_id: int | None
    special_token_ids: tuple[int, ...]
    baseline: TokenSequence
    candidate_span: tuple[int, int]
    variants: tuple[PerturbationVariant, ...]

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ARTIFACT_SCHEMA,
            "generated_at": _iso(self.generated_at),
            "experiment_id": self.experiment_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "policy_id": self.policy_id,
            "prompt_sha256": self.prompt_sha256,
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "mask_token_id": self.mask_token_id,
            "special_token_ids": list(self.special_token_ids),
            "baseline": self.baseline.canonical_dict(),
            "candidate_span": list(self.candidate_span),
            "variants": [variant.canonical_dict() for variant in self.variants],
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PerturbationArtifact:
        _exact_keys(
            values,
            {
                "schema_version",
                "generated_at",
                "experiment_id",
                "model_id",
                "model_revision",
                "tokenizer_id",
                "tokenizer_revision",
                "policy_id",
                "prompt_sha256",
                "vocab_size",
                "pad_token_id",
                "mask_token_id",
                "special_token_ids",
                "baseline",
                "candidate_span",
                "variants",
            },
            "artifact",
        )
        if values["schema_version"] != ARTIFACT_SCHEMA:
            raise EvidenceError("unsupported artifact schema")
        vocab_size = _bounded_int(values["vocab_size"], "vocab_size", minimum=2)
        pad_token_id = _token_id(values["pad_token_id"], "pad_token_id", vocab_size)
        mask_value = values["mask_token_id"]
        mask_token_id = None if mask_value is None else _token_id(mask_value, "mask_token_id", vocab_size)
        if mask_token_id == pad_token_id:
            raise EvidenceError("mask_token_id cannot equal pad_token_id")
        special = _token_ids(values["special_token_ids"], "special_token_ids", vocab_size)
        if len(set(special)) != len(special):
            raise EvidenceError("special_token_ids must be unique")
        baseline = _sequence(values["baseline"], "baseline", vocab_size)
        active = baseline.active(pad_token_id)
        span = _span(values["candidate_span"], "candidate_span", len(active))
        if span[0] == span[1]:
            raise EvidenceError("candidate_span cannot be empty")
        raw_variants = values["variants"]
        if not isinstance(raw_variants, list) or not raw_variants:
            raise EvidenceError("variants must be a non-empty list")
        variants: list[PerturbationVariant] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_variants):
            field = f"variants[{index}]"
            _exact_keys(raw, {"variant_id", "kind", "start", "end", "replacement_ids", "sequence"}, field)
            variant_id = _identifier(raw["variant_id"], f"{field}.variant_id")
            if variant_id in seen:
                raise EvidenceError("variant_id values must be unique")
            seen.add(variant_id)
            kind = raw["kind"]
            if kind not in KINDS:
                raise EvidenceError(f"{field}.kind is unsupported")
            start, end = _span([raw["start"], raw["end"]], field, len(active))
            if start == end:
                raise EvidenceError("perturbation spans cannot be empty")
            replacement = _token_ids(raw["replacement_ids"], f"{field}.replacement_ids", vocab_size)
            variants.append(
                PerturbationVariant(
                    variant_id,
                    kind,
                    start,
                    end,
                    replacement,
                    _sequence(raw["sequence"], f"{field}.sequence", vocab_size),
                )
            )
        digest = values["prompt_sha256"]
        if not isinstance(digest, str) or DIGEST.fullmatch(digest) is None:
            raise EvidenceError("prompt_sha256 must be lowercase SHA-256")
        return cls(
            generated_at=_timestamp(values["generated_at"], "generated_at"),
            experiment_id=_identifier(values["experiment_id"], "experiment_id"),
            model_id=_identifier(values["model_id"], "model_id"),
            model_revision=_revision(values["model_revision"], "model_revision"),
            tokenizer_id=_identifier(values["tokenizer_id"], "tokenizer_id"),
            tokenizer_revision=_revision(values["tokenizer_revision"], "tokenizer_revision"),
            policy_id=_identifier(values["policy_id"], "policy_id"),
            prompt_sha256=digest,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            special_token_ids=tuple(sorted(special)),
            baseline=baseline,
            candidate_span=span,
            variants=tuple(sorted(variants, key=lambda variant: variant.variant_id)),
        )


@dataclass(frozen=True)
class Finding:
    code: str
    message: str
    variant_ref: str | None = None


def audit_artifact(
    artifact: PerturbationArtifact,
    policy: AuditPolicy | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    policy = policy or AuditPolicy()
    artifact = PerturbationArtifact.from_dict(artifact.canonical_dict())
    evaluated_source = now or datetime.now(UTC)
    if evaluated_source.tzinfo is None or evaluated_source.utcoffset() is None:
        raise EvidenceError("evaluation time must include a timezone")
    evaluated_at = evaluated_source.astimezone(UTC)
    findings: list[Finding] = []
    age = evaluated_at - artifact.generated_at
    future_limit = timedelta(seconds=policy.max_future_skew_seconds)
    if age < -future_limit:
        findings.append(Finding("TIME001", "artifact timestamp exceeds allowed future skew"))
    elif age > timedelta(seconds=policy.max_evidence_age_seconds):
        findings.append(Finding("TIME002", "artifact exceeds the evidence-age budget"))
    if len(artifact.variants) > policy.max_variants:
        raise EvidenceError("variant count exceeds policy budget")
    sequences = (artifact.baseline,) + tuple(variant.sequence for variant in artifact.variants)
    if any(len(sequence.input_ids) > policy.max_sequence_tokens for sequence in sequences):
        raise EvidenceError("sequence length exceeds policy budget")
    total_tokens = sum(len(sequence.input_ids) for sequence in sequences)
    if total_tokens > policy.max_total_tokens:
        raise EvidenceError("total token count exceeds policy budget")

    baseline = artifact.baseline.active(artifact.pad_token_id)
    candidate_start, candidate_end = artifact.candidate_span
    special_positions = {index for index, token in enumerate(baseline) if token in artifact.special_token_ids}
    accepted_variants = 0
    code_counts: dict[str, int] = {}
    for variant in artifact.variants:
        before = len(findings)
        ref = _reference(variant.variant_id)
        replacement = variant.replacement_ids
        if variant.kind == "delete" and replacement:
            findings.append(Finding("PERT001", "delete variant supplies replacement tokens", ref))
        if variant.kind == "replace" and not replacement:
            findings.append(Finding("PERT002", "replace variant has no replacement tokens", ref))
        if variant.kind == "mask":
            expected_width = variant.end - variant.start
            if (
                artifact.mask_token_id is None
                or len(replacement) != expected_width
                or any(token != artifact.mask_token_id for token in replacement)
            ):
                findings.append(
                    Finding("PERT003", "mask variant does not use one mask token per target", ref)
                )
        if variant.start < candidate_end and variant.end > candidate_start:
            findings.append(Finding("SCOPE001", "perturbation overlaps the governed candidate span", ref))
        if special_positions.intersection(range(variant.start, variant.end)):
            findings.append(
                Finding("SCOPE002", "perturbation targets a reserved special-token position", ref)
            )
        allowed_replacement = {artifact.mask_token_id} if variant.kind == "mask" else set()
        if any(
            token in artifact.special_token_ids and token not in allowed_replacement for token in replacement
        ):
            findings.append(Finding("SCOPE003", "replacement introduces a reserved special token", ref))

        expected = baseline[: variant.start] + replacement + baseline[variant.end :]
        actual = variant.sequence.active(artifact.pad_token_id)
        if actual != expected:
            findings.append(
                Finding("TOKEN001", "tokenized variant differs outside the declared splice contract", ref)
            )
        if actual == baseline:
            findings.append(Finding("TOKEN002", "perturbation is a token-level no-op", ref))
        if variant.end <= candidate_start:
            delta = len(replacement) - (variant.end - variant.start)
            relocated = (candidate_start + delta, candidate_end + delta)
        else:
            relocated = (candidate_start, candidate_end)
        if (
            relocated[0] < 0
            or relocated[1] > len(actual)
            or actual[slice(*relocated)] != baseline[candidate_start:candidate_end]
        ):
            findings.append(Finding("TOKEN003", "candidate tokens were not preserved exactly", ref))
        if len(findings) == before:
            accepted_variants += 1

    for finding in findings:
        code_counts[finding.code] = code_counts.get(finding.code, 0) + 1
    artifact_sha = _sha256(artifact.canonical_dict())
    bounded = findings[:MAX_REPORTED_FINDINGS]
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": "accepted" if not findings else "rejected",
        "evaluated_at": _iso(evaluated_at),
        "experiment_ref": _reference(artifact.experiment_id),
        "artifact_sha256": artifact_sha,
        "policy": asdict(policy),
        "metrics": {
            "variants": len(artifact.variants),
            "accepted_variants": accepted_variants,
            "rejected_variants": len(artifact.variants) - accepted_variants,
            "total_sequence_tokens": total_tokens,
            "total_findings": len(findings),
            "reported_findings": len(bounded),
            "findings_truncated": len(findings) > len(bounded),
            "finding_code_counts": dict(sorted(code_counts.items())),
        },
        "findings": [asdict(finding) for finding in bounded],
    }
    report["evidence_sha256"] = _sha256(report)
    return report


def _exact_keys(value: Any, expected: set[str], field: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise EvidenceError(f"{field} keys do not match the schema")


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
        raise EvidenceError(f"{field} is not a bounded identifier")
    return value


def _revision(value: Any, field: str) -> str:
    if not isinstance(value, str) or REVISION.fullmatch(value) is None:
        raise EvidenceError(f"{field} must be an immutable 40- or 64-character lowercase hex revision")
    return value


def _bounded_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvidenceError(f"{field} must be an integer >= {minimum}")
    return value


def _token_id(value: Any, field: str, vocab_size: int) -> int:
    token = _bounded_int(value, field)
    if token >= vocab_size:
        raise EvidenceError(f"{field} is outside the tokenizer vocabulary")
    return token


def _token_ids(value: Any, field: str, vocab_size: int) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise EvidenceError(f"{field} must be a list")
    return tuple(_token_id(token, f"{field}[{index}]", vocab_size) for index, token in enumerate(value))


def _span(value: Any, field: str, upper: int) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise EvidenceError(f"{field} must contain start and end")
    start = _bounded_int(value[0], f"{field}[0]")
    end = _bounded_int(value[1], f"{field}[1]")
    if not 0 <= start <= end <= upper:
        raise EvidenceError(f"{field} is outside the active sequence")
    return start, end


def _sequence(value: Any, field: str, vocab_size: int) -> TokenSequence:
    _exact_keys(value, {"input_ids", "attention_mask"}, field)
    input_ids = _token_ids(value["input_ids"], f"{field}.input_ids", vocab_size)
    raw_mask = value["attention_mask"]
    if not isinstance(raw_mask, list) or len(raw_mask) != len(input_ids) or not input_ids:
        raise EvidenceError(f"{field}.attention_mask must align with a non-empty token sequence")
    if any(isinstance(item, bool) or item not in (0, 1) for item in raw_mask):
        raise EvidenceError(f"{field}.attention_mask must contain binary integers")
    return TokenSequence(input_ids, tuple(raw_mask))


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise EvidenceError(f"{field} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise EvidenceError(f"{field} is not valid ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise EvidenceError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def _iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise EvidenceError("timestamps must include a timezone")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _reference(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise EvidenceError(f"non-finite JSON value: {value}")


def load_artifact(path: Path) -> PerturbationArtifact:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise EvidenceError("artifact path must be a regular non-symlink file")
    if metadata.st_size > MAX_INPUT_BYTES:
        raise EvidenceError(f"artifact exceeds the {MAX_INPUT_BYTES}-byte budget")
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise EvidenceError("artifact changed before it could be read")
        payload = handle.read(MAX_INPUT_BYTES + 1)
        final = os.fstat(handle.fileno())
    if len(payload) > MAX_INPUT_BYTES:
        raise EvidenceError(f"artifact exceeds the {MAX_INPUT_BYTES}-byte budget")
    if (opened.st_size, opened.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
        raise EvidenceError("artifact changed while it was being read")
    try:
        values = json.loads(
            payload, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError("artifact is not valid JSON") from error
    if not isinstance(values, dict):
        raise EvidenceError("artifact root must be an object")
    return PerturbationArtifact.from_dict(values)


def _atomic_write(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(values, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit token-level perturbation integrity")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-evidence-age-seconds", type=int, default=3600)
    parser.add_argument("--max-future-skew-seconds", type=int, default=30)
    values = parser.parse_args(argv)
    try:
        policy = AuditPolicy(values.max_evidence_age_seconds, values.max_future_skew_seconds)
        report = audit_artifact(load_artifact(values.input), policy)
        _atomic_write(values.output, report)
    except (EvidenceError, OSError, TypeError, ValueError) as error:
        print(json.dumps({"status": "malformed", "error": str(error)}))
        return 3
    print(json.dumps({"status": report["status"], "evidence_sha256": report["evidence_sha256"]}))
    return 0 if report["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
