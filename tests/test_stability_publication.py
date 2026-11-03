from __future__ import annotations

import hashlib
import json

import pytest

from structxai.stability_lineage import create_manifest_publication_reference
from structxai.stability_publication import (
    MANIFEST_PUBLICATION_VERIFICATION_SCHEMA,
    MAX_MANIFEST_PUBLICATION_RECORD_BYTES,
    StabilityPublicationError,
    create_manifest_publication_record,
    verify_manifest_publication_record,
)
from tests.test_stability_runtime import _fixture

PUBLICATION_URI = "https://evidence.example/struct-xai/public-cohort-v1.json"
PUBLISHED_AT = 1_799_000_000


def _manifest(tmp_path) -> dict[str, object]:
    _reports, manifest, _model, _tokenizer, _examples, _protocol = _fixture(tmp_path)
    return manifest


def _publication(tmp_path, payload: bytes | None = None):
    manifest = _manifest(tmp_path)
    record = payload or create_manifest_publication_record(
        manifest,
        published_at_epoch=PUBLISHED_AT,
    )
    reference = create_manifest_publication_reference(
        publication_uri=PUBLICATION_URI,
        publication_record_sha256=hashlib.sha256(record).hexdigest(),
        manifest_sha256=str(manifest["manifest_sha256"]),
        published_at_epoch=PUBLISHED_AT,
    )
    return manifest, record, reference


def test_published_manifest_record_verifies_before_execution(tmp_path) -> None:
    manifest, record, reference = _publication(tmp_path)

    verification = verify_manifest_publication_record(
        reference,
        record,
        retrieved_uri=PUBLICATION_URI,
    )

    assert verification["schema_version"] == MANIFEST_PUBLICATION_VERIFICATION_SCHEMA
    assert verification["verified"] is True
    assert verification["manifest_sha256"] == manifest["manifest_sha256"]
    assert verification["publication_record_sha256"] == hashlib.sha256(record).hexdigest()
    assert verification["record_bytes"] == len(record)
    assert len(str(verification["verification_sha256"])) == 64
    assert PUBLICATION_URI not in str(verification)
    assert "public-task-00" not in str(verification)


def test_wrong_retrieved_uri_is_rejected_before_record_parsing(tmp_path) -> None:
    _manifest, record, reference = _publication(tmp_path)

    with pytest.raises(StabilityPublicationError, match="retrieved URI"):
        verify_manifest_publication_record(
            reference,
            record,
            retrieved_uri="https://evidence.example/struct-xai/replacement.json",
        )


def test_publication_reference_requires_exact_schema(tmp_path) -> None:
    _manifest, record, reference = _publication(tmp_path)
    reference["unreviewed"] = True

    with pytest.raises(StabilityPublicationError, match="reference is invalid"):
        verify_manifest_publication_record(
            reference,
            record,
            retrieved_uri=PUBLICATION_URI,
        )


def test_downloaded_byte_substitution_is_rejected(tmp_path) -> None:
    _manifest, record, reference = _publication(tmp_path)

    with pytest.raises(StabilityPublicationError, match="downloaded bytes"):
        verify_manifest_publication_record(
            reference,
            record + b" ",
            retrieved_uri=PUBLICATION_URI,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: {**value, "published_at_epoch": PUBLISHED_AT + 1}, "publication epoch"),
        (lambda value: {**value, "published_at_epoch": True}, "record epoch is invalid"),
        (lambda value: {**value, "manifest_sha256": "0" * 64}, "referenced manifest"),
        (lambda value: {**value, "schema_version": "unsupported/v9"}, "schema is unsupported"),
        (lambda value: {**value, "unexpected": True}, "exact schema"),
    ],
)
def test_record_metadata_substitution_is_rejected(tmp_path, mutate, message: str) -> None:
    _manifest, record, reference = _publication(tmp_path)
    document = json.loads(record)
    tampered = json.dumps(
        mutate(document),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    reference["publication_record_sha256"] = hashlib.sha256(tampered).hexdigest()

    with pytest.raises(StabilityPublicationError, match=message):
        verify_manifest_publication_record(
            reference,
            tampered,
            retrieved_uri=PUBLICATION_URI,
        )


def test_nested_manifest_tampering_is_rejected_even_when_record_digest_matches(tmp_path) -> None:
    _manifest, record, reference = _publication(tmp_path)
    document = json.loads(record)
    document["manifest"]["cohort_id"] = "replacement"
    tampered = json.dumps(document, separators=(",", ":"), sort_keys=True).encode("ascii")
    reference["publication_record_sha256"] = hashlib.sha256(tampered).hexdigest()

    with pytest.raises(StabilityPublicationError, match="published stability manifest"):
        verify_manifest_publication_record(
            reference,
            tampered,
            retrieved_uri=PUBLICATION_URI,
        )


def test_duplicate_keys_are_rejected_after_raw_digest_verification(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    duplicate = (
        b'{"manifest":{},"manifest":{},"manifest_sha256":"'
        + str(manifest["manifest_sha256"]).encode("ascii")
        + b'","published_at_epoch":1799000000,'
        + b'"schema_version":"struct-xai-manifest-publication-record/v1"}'
    )
    reference = create_manifest_publication_reference(
        publication_uri=PUBLICATION_URI,
        publication_record_sha256=hashlib.sha256(duplicate).hexdigest(),
        manifest_sha256=str(manifest["manifest_sha256"]),
        published_at_epoch=PUBLISHED_AT,
    )

    with pytest.raises(StabilityPublicationError, match="duplicate keys"):
        verify_manifest_publication_record(
            reference,
            duplicate,
            retrieved_uri=PUBLICATION_URI,
        )


def test_noncanonical_record_is_rejected_even_when_digest_matches(tmp_path) -> None:
    _manifest, record, reference = _publication(tmp_path)
    noncanonical = record + b"\n"
    reference["publication_record_sha256"] = hashlib.sha256(noncanonical).hexdigest()

    with pytest.raises(StabilityPublicationError, match="not canonical"):
        verify_manifest_publication_record(
            reference,
            noncanonical,
            retrieved_uri=PUBLICATION_URI,
        )


@pytest.mark.parametrize("payload", [b"", b"{" + b" " * MAX_MANIFEST_PUBLICATION_RECORD_BYTES])
def test_record_bytes_are_bounded(tmp_path, payload: bytes) -> None:
    manifest = _manifest(tmp_path)
    reference = create_manifest_publication_reference(
        publication_uri=PUBLICATION_URI,
        publication_record_sha256=hashlib.sha256(payload).hexdigest(),
        manifest_sha256=str(manifest["manifest_sha256"]),
        published_at_epoch=PUBLISHED_AT,
    )

    with pytest.raises(StabilityPublicationError, match="size limit"):
        verify_manifest_publication_record(
            reference,
            payload,
            retrieved_uri=PUBLICATION_URI,
        )


@pytest.mark.parametrize("published_at", [True, 0, (1 << 53)])
def test_record_creation_rejects_invalid_publication_epoch(tmp_path, published_at) -> None:
    with pytest.raises(StabilityPublicationError, match="epoch is invalid"):
        create_manifest_publication_record(
            _manifest(tmp_path),
            published_at_epoch=published_at,
        )
