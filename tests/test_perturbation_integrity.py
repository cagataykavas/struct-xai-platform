from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from structxai.perturbation_integrity import (
    ARTIFACT_SCHEMA,
    MAX_INPUT_BYTES,
    AuditPolicy,
    EvidenceError,
    PerturbationArtifact,
    audit_artifact,
    load_artifact,
    main,
)

NOW = datetime(2026, 10, 1, tzinfo=UTC)


def _sequence(active: list[int], *, width: int = 10) -> dict:
    padding = width - len(active)
    return {"input_ids": active + [0] * padding, "attention_mask": [1] * len(active) + [0] * padding}


def _variant(
    variant_id: str = "delete-country",
    *,
    kind: str = "delete",
    start: int = 2,
    end: int = 3,
    replacement_ids: list[int] | None = None,
    active: list[int] | None = None,
) -> dict:
    replacement = [] if replacement_ids is None else replacement_ids
    baseline = [1, 10, 20, 30, 40, 50, 2]
    expected = baseline[:start] + replacement + baseline[end:]
    return {
        "variant_id": variant_id,
        "kind": kind,
        "start": start,
        "end": end,
        "replacement_ids": replacement,
        "sequence": _sequence(active or expected),
    }


def _values(
    *, variants: list[dict] | None = None, generated_at: datetime = NOW - timedelta(seconds=5)
) -> dict:
    return {
        "schema_version": ARTIFACT_SCHEMA,
        "generated_at": generated_at.isoformat(),
        "experiment_id": "turkish-capital-001",
        "model_id": "Qwen-0.5B",
        "model_revision": "b" * 40,
        "tokenizer_id": "Qwen-tokenizer",
        "tokenizer_revision": "c" * 40,
        "policy_id": "perturbation-policy-v1",
        "prompt_sha256": "a" * 64,
        "vocab_size": 100,
        "pad_token_id": 0,
        "mask_token_id": 99,
        "special_token_ids": [1, 2],
        "baseline": _sequence([1, 10, 20, 30, 40, 50, 2]),
        "candidate_span": [4, 6],
        "variants": variants or [_variant()],
    }


def _artifact(**changes) -> PerturbationArtifact:
    values = _values()
    if isinstance(changes.get("generated_at"), datetime):
        changes["generated_at"] = changes["generated_at"].isoformat()
    values.update(changes)
    return PerturbationArtifact.from_dict(values)


def _codes(report: dict) -> set[str]:
    return {finding["code"] for finding in report["findings"]}


@pytest.mark.parametrize(
    "variant",
    [
        _variant(),
        _variant("replace", kind="replace", replacement_ids=[21, 22]),
        _variant("mask", kind="mask", replacement_ids=[99]),
    ],
)
def test_valid_delete_replace_and_mask_are_accepted(variant: dict) -> None:
    report = audit_artifact(PerturbationArtifact.from_dict(_values(variants=[variant])), now=NOW)
    assert report["status"] == "accepted"
    assert report["metrics"]["accepted_variants"] == 1


def test_scope_mutation_is_detected() -> None:
    variant = _variant(active=[1, 11, 30, 40, 50, 2])
    report = audit_artifact(PerturbationArtifact.from_dict(_values(variants=[variant])), now=NOW)
    assert "TOKEN001" in _codes(report)


def test_candidate_overlap_and_special_token_target_are_rejected() -> None:
    candidate = _variant("candidate", start=4, end=5, active=[1, 10, 20, 30, 50, 2])
    special = _variant("special", start=0, end=1, active=[10, 20, 30, 40, 50, 2])
    report = audit_artifact(PerturbationArtifact.from_dict(_values(variants=[candidate, special])), now=NOW)
    assert {"SCOPE001", "SCOPE002"}.issubset(_codes(report))


def test_reserved_special_token_cannot_be_introduced() -> None:
    variant = _variant(kind="replace", replacement_ids=[2])
    report = audit_artifact(PerturbationArtifact.from_dict(_values(variants=[variant])), now=NOW)
    assert "SCOPE003" in _codes(report)


def test_noop_is_rejected() -> None:
    variant = _variant(kind="replace", replacement_ids=[20])
    report = audit_artifact(PerturbationArtifact.from_dict(_values(variants=[variant])), now=NOW)
    assert "TOKEN002" in _codes(report)


@pytest.mark.parametrize(
    "variant,code",
    [
        (_variant(kind="delete", replacement_ids=[77]), "PERT001"),
        (_variant(kind="replace", replacement_ids=[]), "PERT002"),
        (_variant(kind="mask", replacement_ids=[98]), "PERT003"),
    ],
)
def test_kind_contracts_fail_closed(variant: dict, code: str) -> None:
    report = audit_artifact(PerturbationArtifact.from_dict(_values(variants=[variant])), now=NOW)
    assert code in _codes(report)


def test_report_is_deterministic_order_invariant_and_private() -> None:
    first = _variant("b")
    second = _variant("a", start=1, end=2, active=[1, 20, 30, 40, 50, 2])
    forward = audit_artifact(PerturbationArtifact.from_dict(_values(variants=[first, second])), now=NOW)
    reverse = audit_artifact(PerturbationArtifact.from_dict(_values(variants=[second, first])), now=NOW)
    assert forward == reverse
    serialized = json.dumps(forward)
    assert "turkish-capital-001" not in serialized
    assert '"input_ids"' not in serialized
    assert '"replacement_ids"' not in serialized


def test_stale_and_future_artifacts_are_rejected() -> None:
    stale = audit_artifact(_artifact(generated_at=NOW - timedelta(hours=2)), now=NOW)
    future = audit_artifact(_artifact(generated_at=NOW + timedelta(minutes=1)), now=NOW)
    assert "TIME002" in _codes(stale)
    assert "TIME001" in _codes(future)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.update(schema_version="wrong"), "schema"),
        (lambda value: value.update(prompt_sha256="BAD"), "SHA-256"),
        (lambda value: value.update(vocab_size=True), "integer"),
        (lambda value: value.update(model_revision="main"), "immutable"),
        (lambda value: value["baseline"]["input_ids"].__setitem__(0, 100), "vocabulary"),
        (lambda value: value["candidate_span"].__setitem__(1, 99), "active sequence"),
        (lambda value: value.update(extra=True), "keys"),
    ],
)
def test_strict_schema_validation(mutation, match: str) -> None:
    values = _values()
    mutation(values)
    with pytest.raises(EvidenceError, match=match):
        PerturbationArtifact.from_dict(values)


def test_padding_and_attention_mask_must_be_canonical() -> None:
    values = _values()
    values["baseline"]["attention_mask"] = [1, 0, 1, 1, 1, 1, 1, 0, 0, 0]
    with pytest.raises(EvidenceError, match="contiguous"):
        PerturbationArtifact.from_dict(values)
    values = _values()
    values["baseline"]["input_ids"][-1] = 8
    with pytest.raises(EvidenceError, match="pad_token_id"):
        PerturbationArtifact.from_dict(values)
    values = _values()
    values["baseline"]["input_ids"][1] = 0
    with pytest.raises(EvidenceError, match="active positions"):
        PerturbationArtifact.from_dict(values)
    values = _values()
    values["mask_token_id"] = 0
    with pytest.raises(EvidenceError, match="cannot equal"):
        PerturbationArtifact.from_dict(values)


def test_duplicate_variant_and_special_ids_are_rejected() -> None:
    values = _values(variants=[_variant("same"), _variant("same")])
    with pytest.raises(EvidenceError, match="unique"):
        PerturbationArtifact.from_dict(values)
    values = _values()
    values["special_token_ids"] = [1, 1]
    with pytest.raises(EvidenceError, match="unique"):
        PerturbationArtifact.from_dict(values)


def test_policy_budgets_are_enforced() -> None:
    artifact = _artifact()
    with pytest.raises(EvidenceError, match="variant count"):
        two_variants = PerturbationArtifact.from_dict(
            _values(variants=[_variant("first"), _variant("second")])
        )
        audit_artifact(two_variants, AuditPolicy(max_variants=1), now=NOW)
    with pytest.raises(EvidenceError, match="sequence length"):
        audit_artifact(artifact, AuditPolicy(max_sequence_tokens=5), now=NOW)
    with pytest.raises(EvidenceError, match="total token"):
        audit_artifact(artifact, AuditPolicy(max_total_tokens=15), now=NOW)


def test_finding_report_is_bounded() -> None:
    variants = [
        _variant(f"bad-{index}", kind="delete", start=4, end=5, replacement_ids=[40]) for index in range(1024)
    ]
    report = audit_artifact(PerturbationArtifact.from_dict(_values(variants=variants)), now=NOW)
    assert report["metrics"]["total_findings"] == 3 * 1024
    assert report["metrics"]["reported_findings"] == 2048
    assert report["metrics"]["findings_truncated"] is True


def test_loader_rejects_duplicate_nonfinite_oversized_and_symlink(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":"a","schema_version":"b"}', encoding="utf-8")
    with pytest.raises(EvidenceError, match="duplicate"):
        load_artifact(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(EvidenceError, match="non-finite"):
        load_artifact(nonfinite)
    oversized = tmp_path / "oversized.json"
    oversized.write_text("x" * (MAX_INPUT_BYTES + 1), encoding="utf-8")
    with pytest.raises(EvidenceError, match="budget"):
        load_artifact(oversized)
    valid = tmp_path / "valid.json"
    link = tmp_path / "link.json"
    valid.write_text(json.dumps(_values()), encoding="utf-8")
    link.symlink_to(valid)
    with pytest.raises(EvidenceError, match="non-symlink"):
        load_artifact(link)


class _FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW if tz is not None else NOW.replace(tzinfo=None)


def test_cli_has_distinct_accept_reject_and_malformed_exits(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("structxai.perturbation_integrity.datetime", _FixedDateTime)
    accepted = tmp_path / "accepted.json"
    rejected = tmp_path / "rejected.json"
    malformed = tmp_path / "malformed.json"
    accepted.write_text(json.dumps(_values()), encoding="utf-8")
    rejected.write_text(
        json.dumps(_values(variants=[_variant(kind="replace", replacement_ids=[20])])), encoding="utf-8"
    )
    malformed.write_text("{", encoding="utf-8")
    assert main(["--input", str(accepted), "--output", str(tmp_path / "a.json")]) == 0
    assert main(["--input", str(rejected), "--output", str(tmp_path / "r.json")]) == 2
    assert main(["--input", str(malformed), "--output", str(tmp_path / "m.json")]) == 3
    assert json.loads((tmp_path / "a.json").read_text())["status"] == "accepted"


def test_policy_and_timestamp_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        AuditPolicy(max_variants=0)
    values = _values()
    values["generated_at"] = "2026-10-01T00:00:00"
    with pytest.raises(EvidenceError, match="timezone"):
        PerturbationArtifact.from_dict(values)
