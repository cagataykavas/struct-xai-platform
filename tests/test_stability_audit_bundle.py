from __future__ import annotations

import copy
import hashlib

import pytest

from structxai.stability_audit_bundle import (
    STABILITY_AUDIT_RECEIPT_SCHEMA,
    StabilityAuditBundleError,
    serialize_stability_audit_receipt,
    verify_stability_audit_bundle,
)
from structxai.stability_lineage import (
    create_manifest_publication_reference,
    create_runtime_bound_stability_evidence,
)
from structxai.stability_publication import create_manifest_publication_record
from tests.test_stability_lineage import _CountingAdapter, _runtime_run

PUBLICATION_URI = "https://evidence.example/struct-xai/public-cohort-v1.json"
PUBLISHED_AT = 1_799_000_000


def _audit_fixture(tmp_path):
    manifest, reports, runtime_evidence = _runtime_run(tmp_path)
    record = create_manifest_publication_record(
        manifest,
        published_at_epoch=PUBLISHED_AT,
    )
    reference = create_manifest_publication_reference(
        publication_uri=PUBLICATION_URI,
        publication_record_sha256=hashlib.sha256(record).hexdigest(),
        manifest_sha256=str(manifest["manifest_sha256"]),
        published_at_epoch=PUBLISHED_AT,
    )
    adapter = _CountingAdapter()
    lineage = create_runtime_bound_stability_evidence(
        manifest,
        reports,
        runtime_evidence,
        reference,
        signer=adapter,
    )
    return manifest, record, reference, lineage, adapter


def test_complete_audit_bundle_is_cross_bound_deterministic_and_privacy_reduced(
    tmp_path,
) -> None:
    manifest, record, reference, lineage, adapter = _audit_fixture(tmp_path)

    first = verify_stability_audit_bundle(
        reference,
        record,
        retrieved_uri=PUBLICATION_URI,
        lineage_bundle=lineage,
        verifiers={adapter.key_id: adapter},
    )
    second = verify_stability_audit_bundle(
        reference,
        record,
        retrieved_uri=PUBLICATION_URI,
        lineage_bundle=lineage,
        verifiers={adapter.key_id: adapter},
    )
    payload = serialize_stability_audit_receipt(first)

    assert first == second
    assert first["schema_version"] == STABILITY_AUDIT_RECEIPT_SCHEMA
    assert first["verified"] is True
    assert first["task_count"] == len(manifest["tasks"])
    assert first["manifest_sha256"] == manifest["manifest_sha256"]
    assert first["publication_record_sha256"] == hashlib.sha256(record).hexdigest()
    assert len(str(first["receipt_sha256"])) == 64
    assert PUBLICATION_URI.encode("ascii") not in payload
    assert str(manifest["cohort_id"]).encode("ascii") not in payload
    assert adapter.key_id.encode("ascii") not in payload
    assert b"public-task-00" not in payload


def test_independently_valid_publication_cannot_be_mixed_with_other_lineage(
    tmp_path,
) -> None:
    _manifest, record, reference, lineage, adapter = _audit_fixture(tmp_path)
    replacement_uri = "https://mirror.example/struct-xai/public-cohort-v1.json"
    replacement_reference = create_manifest_publication_reference(
        publication_uri=replacement_uri,
        publication_record_sha256=str(reference["publication_record_sha256"]),
        manifest_sha256=str(reference["manifest_sha256"]),
        published_at_epoch=PUBLISHED_AT,
    )

    with pytest.raises(StabilityAuditBundleError, match="do not cross-bind"):
        verify_stability_audit_bundle(
            replacement_reference,
            record,
            retrieved_uri=replacement_uri,
            lineage_bundle=lineage,
            verifiers={adapter.key_id: adapter},
        )


def test_publication_failure_is_sanitized_at_bundle_boundary(tmp_path) -> None:
    _manifest, record, reference, lineage, adapter = _audit_fixture(tmp_path)

    with pytest.raises(StabilityAuditBundleError, match="publication verification failed") as error:
        verify_stability_audit_bundle(
            reference,
            record + b"private upstream diagnostic",
            retrieved_uri=PUBLICATION_URI,
            lineage_bundle=lineage,
            verifiers={adapter.key_id: adapter},
        )

    assert "private upstream diagnostic" not in str(error.value)


def test_lineage_failure_is_sanitized_at_bundle_boundary(tmp_path) -> None:
    _manifest, record, reference, lineage, adapter = _audit_fixture(tmp_path)
    tampered = copy.deepcopy(lineage)
    tampered["signature"] = tampered["signature"][:-1] + ("A" if tampered["signature"][-1] != "A" else "B")

    with pytest.raises(StabilityAuditBundleError, match="runtime lineage verification failed"):
        verify_stability_audit_bundle(
            reference,
            record,
            retrieved_uri=PUBLICATION_URI,
            lineage_bundle=tampered,
            verifiers={adapter.key_id: adapter},
        )


def test_retained_receipt_rejects_tampering_and_schema_extension(tmp_path) -> None:
    _manifest, record, reference, lineage, adapter = _audit_fixture(tmp_path)
    receipt = verify_stability_audit_bundle(
        reference,
        record,
        retrieved_uri=PUBLICATION_URI,
        lineage_bundle=lineage,
        verifiers={adapter.key_id: adapter},
    )

    tampered = {**receipt, "task_count": int(receipt["task_count"]) + 1}
    with pytest.raises(StabilityAuditBundleError, match="digest does not match"):
        serialize_stability_audit_receipt(tampered)

    with pytest.raises(StabilityAuditBundleError, match="exact schema"):
        serialize_stability_audit_receipt({**receipt, "unreviewed": True})


def test_bundle_requires_known_lineage_verifier(tmp_path) -> None:
    _manifest, record, reference, lineage, _adapter = _audit_fixture(tmp_path)

    with pytest.raises(StabilityAuditBundleError, match="runtime lineage verification failed"):
        verify_stability_audit_bundle(
            reference,
            record,
            retrieved_uri=PUBLICATION_URI,
            lineage_bundle=lineage,
            verifiers={},
        )
