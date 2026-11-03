from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping

from structxai.stability_lineage import create_manifest_publication_reference
from structxai.stability_manifest import (
    StabilityTaskManifestError,
    validate_stability_task_manifest,
)

MANIFEST_PUBLICATION_RECORD_SCHEMA = "struct-xai-manifest-publication-record/v1"
MANIFEST_PUBLICATION_VERIFICATION_SCHEMA = "struct-xai-manifest-publication-verification/v1"
MAX_MANIFEST_PUBLICATION_RECORD_BYTES = 512 * 1024

_RECORD_FIELDS = {
    "schema_version",
    "published_at_epoch",
    "manifest_sha256",
    "manifest",
}
_REFERENCE_FIELDS = {
    "schema_version",
    "publication_uri",
    "publication_record_sha256",
    "manifest_sha256",
    "published_at_epoch",
}


class StabilityPublicationError(ValueError):
    """Stable rejection from the external manifest-publication boundary."""

    code = "STABILITY_PUBLICATION_REJECTED"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise StabilityPublicationError("publication record is not canonical JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_document(value: object) -> str:
    return _sha256_bytes(_canonical_json(value))


def _same(left: object, right: object) -> bool:
    return isinstance(left, str) and isinstance(right, str) and hmac.compare_digest(left, right)


def _normalized_reference(reference: object) -> dict[str, object]:
    if not isinstance(reference, Mapping) or set(reference) != _REFERENCE_FIELDS:
        raise StabilityPublicationError("manifest publication reference is invalid")
    try:
        return create_manifest_publication_reference(
            publication_uri=reference["publication_uri"],
            publication_record_sha256=reference["publication_record_sha256"],
            manifest_sha256=reference["manifest_sha256"],
            published_at_epoch=reference["published_at_epoch"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StabilityPublicationError("manifest publication reference is invalid") from exc


def create_manifest_publication_record(
    manifest: Mapping[str, object],
    *,
    published_at_epoch: int,
) -> bytes:
    """Create the exact canonical bytes that should be retained before evaluation."""

    try:
        normalized_manifest = validate_stability_task_manifest(dict(manifest))
    except (TypeError, ValueError, StabilityTaskManifestError) as exc:
        raise StabilityPublicationError("stability manifest is invalid") from exc
    if (
        isinstance(published_at_epoch, bool)
        or not isinstance(published_at_epoch, int)
        or not 1 <= published_at_epoch <= (1 << 53) - 1
    ):
        raise StabilityPublicationError("publication epoch is invalid")
    record = {
        "schema_version": MANIFEST_PUBLICATION_RECORD_SCHEMA,
        "published_at_epoch": published_at_epoch,
        "manifest_sha256": normalized_manifest["manifest_sha256"],
        "manifest": normalized_manifest,
    }
    payload = _canonical_json(record)
    if len(payload) > MAX_MANIFEST_PUBLICATION_RECORD_BYTES:
        raise StabilityPublicationError("publication record exceeds the size limit")
    return payload


def _load_record(payload: bytes) -> dict[str, object]:
    if not isinstance(payload, bytes) or not 1 <= len(payload) <= MAX_MANIFEST_PUBLICATION_RECORD_BYTES:
        raise StabilityPublicationError("publication record is empty or exceeds the size limit")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise StabilityPublicationError("publication record contains duplicate keys")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise StabilityPublicationError(f"publication record contains invalid constant: {value}")

    try:
        document = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StabilityPublicationError("publication record is malformed") from exc
    if not isinstance(document, dict) or set(document) != _RECORD_FIELDS:
        raise StabilityPublicationError("publication record must use the exact schema")
    if payload != _canonical_json(document):
        raise StabilityPublicationError("publication record is not canonical JSON")
    return document


def verify_manifest_publication_record(
    reference: Mapping[str, object],
    record_bytes: bytes,
    *,
    retrieved_uri: str,
) -> dict[str, object]:
    """Verify exact downloaded bytes before a manifest-locked cohort is executed."""

    normalized_reference = _normalized_reference(reference)
    if not isinstance(retrieved_uri, str) or not hmac.compare_digest(
        retrieved_uri,
        str(normalized_reference["publication_uri"]),
    ):
        raise StabilityPublicationError("retrieved URI does not match the publication reference")
    if not isinstance(record_bytes, bytes):
        raise StabilityPublicationError("publication record is invalid")
    if not 1 <= len(record_bytes) <= MAX_MANIFEST_PUBLICATION_RECORD_BYTES:
        raise StabilityPublicationError("publication record is empty or exceeds the size limit")
    record_sha256 = _sha256_bytes(record_bytes)
    if not _same(record_sha256, normalized_reference["publication_record_sha256"]):
        raise StabilityPublicationError("publication record digest does not match downloaded bytes")
    document = _load_record(record_bytes)
    if document.get("schema_version") != MANIFEST_PUBLICATION_RECORD_SCHEMA:
        raise StabilityPublicationError("publication record schema is unsupported")
    record_epoch = document.get("published_at_epoch")
    if (
        isinstance(record_epoch, bool)
        or not isinstance(record_epoch, int)
        or not 1 <= record_epoch <= (1 << 53) - 1
    ):
        raise StabilityPublicationError("publication record epoch is invalid")
    if record_epoch != normalized_reference["published_at_epoch"]:
        raise StabilityPublicationError("publication epoch does not match the reference")
    if not _same(document.get("manifest_sha256"), normalized_reference["manifest_sha256"]):
        raise StabilityPublicationError("publication record does not match the referenced manifest")
    manifest = document.get("manifest")
    try:
        normalized_manifest = validate_stability_task_manifest(manifest)
    except (TypeError, StabilityTaskManifestError) as exc:
        raise StabilityPublicationError("published stability manifest is invalid") from exc
    if not _same(normalized_manifest["manifest_sha256"], document["manifest_sha256"]):
        raise StabilityPublicationError("published manifest digest does not match its record")

    verification = {
        "schema_version": MANIFEST_PUBLICATION_VERIFICATION_SCHEMA,
        "verified": True,
        "publication_uri_sha256": hashlib.sha256(retrieved_uri.encode("ascii")).hexdigest(),
        "publication_record_sha256": record_sha256,
        "manifest_sha256": normalized_manifest["manifest_sha256"],
        "published_at_epoch": document["published_at_epoch"],
        "publication_reference_sha256": _sha256_document(normalized_reference),
        "record_bytes": len(record_bytes),
    }
    verification["verification_sha256"] = _sha256_document(verification)
    return verification
