from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping

from structxai.stability_lineage import (
    StabilityLineageError,
    verify_runtime_bound_stability_evidence,
)
from structxai.stability_manifest import EXTERNAL_SIGNATURE_ALGORITHMS, ExternalEvidenceVerifier
from structxai.stability_publication import (
    StabilityPublicationError,
    verify_manifest_publication_record,
)

STABILITY_AUDIT_RECEIPT_SCHEMA = "struct-xai-stability-audit-receipt/v1"
MAX_STABILITY_AUDIT_RECEIPT_BYTES = 16 * 1024

_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_RECEIPT_FIELDS = {
    "schema_version",
    "verified",
    "task_count",
    "manifest_sha256",
    "publication_record_sha256",
    "publication_reference_sha256",
    "publication_verification_sha256",
    "runtime_evidence_sha256",
    "report_commitments_sha256",
    "cohort_evidence_sha256",
    "signed_evidence_sha256",
    "lineage_verification_sha256",
    "cohort_id_sha256",
    "algorithm",
    "key_id_sha256",
    "key_revision_sha256",
    "receipt_sha256",
}
_DIGEST_FIELDS = _RECEIPT_FIELDS - {
    "schema_version",
    "verified",
    "task_count",
    "algorithm",
}


class StabilityAuditBundleError(ValueError):
    """Stable failure from the end-to-end retained-evidence verification boundary."""

    code = "STABILITY_AUDIT_BUNDLE_REJECTED"

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
        raise StabilityAuditBundleError("audit receipt is not canonical JSON") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _same(left: object, right: object) -> bool:
    return isinstance(left, str) and isinstance(right, str) and hmac.compare_digest(left, right)


def _require_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise StabilityAuditBundleError(f"{name} is invalid")
    return value


def _ascii_sha256(value: str, name: str) -> str:
    try:
        return hashlib.sha256(value.encode("ascii")).hexdigest()
    except UnicodeEncodeError as exc:
        raise StabilityAuditBundleError(f"{name} is invalid") from exc


def _binding_matches(left: object, right: object) -> bool:
    if isinstance(left, str) and isinstance(right, str):
        return hmac.compare_digest(left, right)
    return type(left) is int and type(right) is int and left == right


def verify_stability_audit_bundle(
    publication_reference: Mapping[str, object],
    publication_record_bytes: bytes,
    *,
    retrieved_uri: str,
    lineage_bundle: Mapping[str, object],
    verifiers: Mapping[str, ExternalEvidenceVerifier],
) -> dict[str, object]:
    """Verify publication and runtime lineage together before evidence retention."""

    try:
        publication = verify_manifest_publication_record(
            publication_reference,
            publication_record_bytes,
            retrieved_uri=retrieved_uri,
        )
    except (StabilityPublicationError, TypeError, ValueError) as exc:
        raise StabilityAuditBundleError("publication verification failed") from exc
    try:
        lineage = verify_runtime_bound_stability_evidence(lineage_bundle, verifiers)
    except (StabilityLineageError, TypeError, ValueError) as exc:
        raise StabilityAuditBundleError("runtime lineage verification failed") from exc

    if publication.get("verified") is not True or lineage.get("verified") is not True:
        raise StabilityAuditBundleError("component verification is incomplete")
    bindings = (
        (publication.get("manifest_sha256"), lineage.get("manifest_sha256")),
        (
            publication.get("publication_reference_sha256"),
            lineage.get("manifest_publication_reference_sha256"),
        ),
        (
            publication.get("publication_record_sha256"),
            lineage.get("publication_record_sha256"),
        ),
        (publication.get("published_at_epoch"), lineage.get("published_at_epoch")),
    )
    if any(not _binding_matches(left, right) for left, right in bindings):
        raise StabilityAuditBundleError("publication and runtime lineage do not cross-bind")

    digest_fields = {
        "manifest_sha256": publication.get("manifest_sha256"),
        "publication_record_sha256": publication.get("publication_record_sha256"),
        "publication_reference_sha256": publication.get("publication_reference_sha256"),
        "publication_verification_sha256": publication.get("verification_sha256"),
        "runtime_evidence_sha256": lineage.get("runtime_evidence_sha256"),
        "report_commitments_sha256": lineage.get("report_commitments_sha256"),
        "cohort_evidence_sha256": lineage.get("cohort_evidence_sha256"),
        "signed_evidence_sha256": lineage.get("signed_evidence_sha256"),
        "key_revision_sha256": lineage.get("key_revision_sha256"),
    }
    normalized_digests = {name: _require_digest(value, name) for name, value in digest_fields.items()}
    task_count = lineage.get("task_count")
    algorithm = lineage.get("algorithm")
    cohort_id = lineage.get("cohort_id")
    key_id = lineage.get("key_id")
    if (
        type(task_count) is not int
        or not 1 <= task_count <= 10_000
        or not isinstance(algorithm, str)
        or algorithm not in EXTERNAL_SIGNATURE_ALGORITHMS
        or not isinstance(cohort_id, str)
        or not cohort_id
        or not isinstance(key_id, str)
        or not key_id
    ):
        raise StabilityAuditBundleError("runtime lineage metadata is invalid")
    lineage_verification_sha256 = _digest(lineage)
    receipt: dict[str, object] = {
        "schema_version": STABILITY_AUDIT_RECEIPT_SCHEMA,
        "verified": True,
        "task_count": task_count,
        **normalized_digests,
        "lineage_verification_sha256": lineage_verification_sha256,
        "cohort_id_sha256": _ascii_sha256(cohort_id, "cohort_id"),
        "algorithm": algorithm,
        "key_id_sha256": _ascii_sha256(key_id, "key_id"),
    }
    receipt["receipt_sha256"] = _digest(receipt)
    return receipt


def serialize_stability_audit_receipt(receipt: Mapping[str, object]) -> bytes:
    """Serialize a complete receipt for immutable retention after verification."""

    if not isinstance(receipt, Mapping) or set(receipt) != _RECEIPT_FIELDS:
        raise StabilityAuditBundleError("audit receipt must use the exact schema")
    document = dict(receipt)
    if (
        document.get("schema_version") != STABILITY_AUDIT_RECEIPT_SCHEMA
        or document.get("verified") is not True
        or type(document.get("task_count")) is not int
        or not 1 <= int(document["task_count"]) <= 10_000
        or not isinstance(document.get("algorithm"), str)
        or document["algorithm"] not in EXTERNAL_SIGNATURE_ALGORITHMS
    ):
        raise StabilityAuditBundleError("audit receipt metadata is invalid")
    for field in _DIGEST_FIELDS:
        _require_digest(document.get(field), field)
    claimed = document.pop("receipt_sha256")
    if not _same(claimed, _digest(document)):
        raise StabilityAuditBundleError("audit receipt digest does not match its content")
    document["receipt_sha256"] = claimed
    payload = _canonical_json(document)
    if len(payload) > MAX_STABILITY_AUDIT_RECEIPT_BYTES:
        raise StabilityAuditBundleError("audit receipt exceeds the size limit")
    return payload
