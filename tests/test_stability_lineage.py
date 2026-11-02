from __future__ import annotations

import copy
import hashlib

import pytest

from structxai.stability import audit_artifact_stability_cohort
from structxai.stability_lineage import (
    StabilityLineageError,
    create_manifest_publication_reference,
    create_runtime_bound_stability_evidence,
    verify_runtime_bound_stability_evidence,
)
from structxai.stability_runtime import run_manifest_locked_stability_cohort
from tests.test_stability_manifest import _artifact, _TestExternalAdapter
from tests.test_stability_runtime import _fixture


class _CountingAdapter(_TestExternalAdapter):
    def __init__(self, secret: bytes = b"test-only-lineage-key") -> None:
        super().__init__(secret)
        self.sign_calls = 0

    def sign(self, message: bytes) -> bytes:
        self.sign_calls += 1
        return super().sign(message)


class _FailSecondSigner(_CountingAdapter):
    def sign(self, message: bytes) -> bytes:
        self.sign_calls += 1
        if self.sign_calls == 2:
            raise RuntimeError("private provider diagnostic")
        return hashlib.sha512(self._secret + message).digest()


def _runtime_run(tmp_path, reports_override=None):
    reports, manifest, model, tokenizer, examples, protocol = _fixture(tmp_path)
    active_reports = reports_override or reports
    returned, evidence = run_manifest_locked_stability_cohort(
        manifest,
        expected_manifest_sha256=manifest["manifest_sha256"],
        model_path=model,
        tokenizer_path=tokenizer,
        dataset_examples=examples,
        selection_protocol=protocol,
        runner=lambda inputs: active_reports,
    )
    return manifest, returned, evidence


def _alternate_reports() -> dict[str, dict]:
    return {
        f"public-task-{task_index:02d}": audit_artifact_stability_cohort(
            {f"seed-{seed}": _artifact(task_index, seed) for seed in range(3, 6)}
        )
        for task_index in range(8)
    }


def _publication(manifest: dict[str, object]) -> dict[str, object]:
    return create_manifest_publication_reference(
        publication_uri="https://evidence.example/struct-xai/public-cohort-v1.json",
        publication_record_sha256=hashlib.sha256(b"public-manifest-record-v1").hexdigest(),
        manifest_sha256=str(manifest["manifest_sha256"]),
        published_at_epoch=1_799_000_000,
    )


def test_lineage_envelope_verifies_runtime_and_signed_cohort_as_one_unit(tmp_path) -> None:
    manifest, reports, runtime_evidence = _runtime_run(tmp_path)
    adapter = _CountingAdapter()

    bundle = create_runtime_bound_stability_evidence(
        manifest,
        reports,
        runtime_evidence,
        _publication(manifest),
        signer=adapter,
    )
    verification = verify_runtime_bound_stability_evidence(
        bundle,
        {adapter.key_id: adapter},
    )

    assert adapter.sign_calls == 2
    assert verification["verified"] is True
    assert verification["manifest_sha256"] == manifest["manifest_sha256"]
    assert (
        verification["manifest_publication_reference_sha256"]
        == bundle["manifest_publication_reference_sha256"]
    )
    assert (
        verification["publication_record_sha256"] == hashlib.sha256(b"public-manifest-record-v1").hexdigest()
    )
    assert verification["runtime_evidence_sha256"] == runtime_evidence.evidence_sha256
    assert verification["report_commitments_sha256"] == (runtime_evidence.report_commitments_sha256)
    assert "record-0" not in str(bundle)
    assert "Private public-task rendering" not in str(bundle)


def test_lineage_creation_rejects_valid_reports_from_a_different_runtime_run(tmp_path) -> None:
    manifest, _, runtime_evidence = _runtime_run(tmp_path)
    alternate_reports = _alternate_reports()
    assert all(report["accepted"] for report in alternate_reports.values())
    adapter = _CountingAdapter()

    with pytest.raises(StabilityLineageError, match="does not commit to the supplied reports"):
        create_runtime_bound_stability_evidence(
            manifest,
            alternate_reports,
            runtime_evidence,
            _publication(manifest),
            signer=adapter,
        )

    assert adapter.sign_calls == 0


def test_lineage_creation_rejects_runtime_evidence_from_a_different_report_set(tmp_path) -> None:
    manifest, reports, _ = _runtime_run(tmp_path / "first")
    _, _, alternate_runtime = _runtime_run(
        tmp_path / "second",
        reports_override=_alternate_reports(),
    )

    with pytest.raises(StabilityLineageError, match="does not commit to the supplied reports"):
        create_runtime_bound_stability_evidence(
            manifest,
            reports,
            alternate_runtime,
            _publication(manifest),
            signer=_CountingAdapter(),
        )


def test_lineage_rejects_tampered_runtime_input_and_evidence_digests(tmp_path) -> None:
    manifest, reports, runtime_evidence = _runtime_run(tmp_path)
    runtime_document = runtime_evidence.to_audit_dict()
    runtime_document["model_weights_sha256"] = "0" * 64

    with pytest.raises(StabilityLineageError, match="input commitments"):
        create_runtime_bound_stability_evidence(
            manifest,
            reports,
            runtime_document,
            _publication(manifest),
            signer=_CountingAdapter(),
        )

    runtime_document = runtime_evidence.to_audit_dict()
    runtime_document["evidence_sha256"] = "0" * 64
    with pytest.raises(StabilityLineageError, match="digest does not match"):
        create_runtime_bound_stability_evidence(
            manifest,
            reports,
            runtime_document,
            _publication(manifest),
            signer=_CountingAdapter(),
        )


@pytest.mark.parametrize("target", ["signed_evidence", "runtime_evidence"])
def test_verification_rejects_nested_artifact_tampering(tmp_path, target: str) -> None:
    manifest, reports, runtime_evidence = _runtime_run(tmp_path)
    adapter = _CountingAdapter()
    bundle = create_runtime_bound_stability_evidence(
        manifest,
        reports,
        runtime_evidence,
        _publication(manifest),
        signer=adapter,
    )
    tampered = copy.deepcopy(bundle)
    if target == "signed_evidence":
        tampered[target]["task_count"] = 7
    else:
        tampered[target]["postflight_verified"] = False

    with pytest.raises(StabilityLineageError, match="signed cohort|runtime evidence"):
        verify_runtime_bound_stability_evidence(tampered, {adapter.key_id: adapter})


def test_verification_rejects_outer_digest_claim_and_signature_tampering(tmp_path) -> None:
    manifest, reports, runtime_evidence = _runtime_run(tmp_path)
    adapter = _CountingAdapter()
    bundle = create_runtime_bound_stability_evidence(
        manifest,
        reports,
        runtime_evidence,
        _publication(manifest),
        signer=adapter,
    )

    tampered = copy.deepcopy(bundle)
    tampered["report_commitments_sha256"] = "0" * 64
    with pytest.raises(StabilityLineageError, match="digest claims"):
        verify_runtime_bound_stability_evidence(tampered, {adapter.key_id: adapter})

    tampered = copy.deepcopy(bundle)
    tampered["signature"] = tampered["signature"][:-1] + ("A" if tampered["signature"][-1] != "A" else "B")
    with pytest.raises(StabilityLineageError, match="signature is invalid"):
        verify_runtime_bound_stability_evidence(tampered, {adapter.key_id: adapter})


def test_lineage_requires_exact_schema_and_known_key(tmp_path) -> None:
    manifest, reports, runtime_evidence = _runtime_run(tmp_path)
    adapter = _CountingAdapter()
    bundle = create_runtime_bound_stability_evidence(
        manifest,
        reports,
        runtime_evidence,
        _publication(manifest),
        signer=adapter,
    )

    unexpected = copy.deepcopy(bundle)
    unexpected["unexpected"] = True
    with pytest.raises(StabilityLineageError, match="exact schema"):
        verify_runtime_bound_stability_evidence(unexpected, {adapter.key_id: adapter})
    with pytest.raises(StabilityLineageError, match="signed cohort evidence is invalid"):
        verify_runtime_bound_stability_evidence(bundle, {})


def test_lineage_signing_failure_is_sanitized(tmp_path) -> None:
    manifest, reports, runtime_evidence = _runtime_run(tmp_path)

    with pytest.raises(StabilityLineageError, match="external lineage signing failed") as error:
        create_runtime_bound_stability_evidence(
            manifest,
            reports,
            runtime_evidence,
            _publication(manifest),
            signer=_FailSecondSigner(),
        )

    assert "private provider diagnostic" not in str(error.value)


def test_verification_rejects_publication_reference_substitution(tmp_path) -> None:
    manifest, reports, runtime_evidence = _runtime_run(tmp_path)
    adapter = _CountingAdapter()
    bundle = create_runtime_bound_stability_evidence(
        manifest,
        reports,
        runtime_evidence,
        _publication(manifest),
        signer=adapter,
    )

    tampered = copy.deepcopy(bundle)
    tampered["manifest_publication_reference"]["publication_uri"] = (
        "https://evidence.example/struct-xai/replacement.json"
    )

    with pytest.raises(StabilityLineageError, match="digest claims"):
        verify_runtime_bound_stability_evidence(tampered, {adapter.key_id: adapter})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("publication_uri", "http://evidence.example/manifest.json", "URI is invalid"),
        (
            "publication_uri",
            "https://user:secret@evidence.example/manifest.json",
            "URI is invalid",
        ),
        (
            "publication_uri",
            "https://evidence.example/manifest.json?mutable=yes",
            "URI is invalid",
        ),
        (
            "publication_uri",
            "https://evidence.example/manifest.json\n",
            "URI is invalid",
        ),
        ("manifest_sha256", "0" * 64, "does not match the manifest"),
        ("publication_record_sha256", "not-a-digest", "record digest is invalid"),
        ("published_at_epoch", True, "epoch is invalid"),
    ],
)
def test_creation_rejects_invalid_publication_reference_before_signing(
    tmp_path,
    field: str,
    value: object,
    message: str,
) -> None:
    manifest, reports, runtime_evidence = _runtime_run(tmp_path)
    reference = _publication(manifest)
    reference[field] = value
    adapter = _CountingAdapter()

    with pytest.raises(StabilityLineageError, match=message):
        create_runtime_bound_stability_evidence(
            manifest,
            reports,
            runtime_evidence,
            reference,
            signer=adapter,
        )

    assert adapter.sign_calls == 0
