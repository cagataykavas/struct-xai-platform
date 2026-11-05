from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
from collections.abc import Mapping

from structxai.stability_audit_bundle import (
    StabilityAuditBundleError,
    verify_stability_audit_bundle,
)
from structxai.stability_lineage import (
    MANIFEST_PUBLICATION_REFERENCE_SCHEMA,
    StabilityLineageError,
    create_manifest_publication_reference,
)
from structxai.stability_manifest import (
    EXTERNAL_SIGNATURE_ALGORITHMS,
    EXTERNAL_SIGNATURE_ENCODING,
    MAX_EXTERNAL_SIGNATURE_BYTES,
    MIN_EXTERNAL_SIGNATURE_BYTES,
    ExternalEvidenceSigner,
    ExternalEvidenceVerifier,
)

PUBLICATION_REFERENCE_CUSTODY_SCHEMA = "struct-xai-publication-reference-custody/v2"
PUBLICATION_REFERENCE_CUSTODY_PURPOSE = "independent-publication-reference-custody"
PUBLICATION_REFERENCE_CUSTODY_VERIFICATION_SCHEMA = "struct-xai-publication-reference-custody-verification/v2"
PUBLICATION_REFERENCE_CUSTODY_REVOCATION_VERIFICATION_SCHEMA = (
    "struct-xai-publication-reference-custody-revocation-verification/v1"
)
CUSTODIED_STABILITY_AUDIT_RECEIPT_SCHEMA = "struct-xai-custodied-stability-audit-receipt/v2"
CUSTODY_KEY_STATUS_SNAPSHOT_SCHEMA = "struct-xai-custody-key-status-snapshot/v2"
CUSTODY_KEY_STATUS_VERIFICATION_SCHEMA = "struct-xai-custody-key-status-verification/v2"
CUSTODY_KEY_STATUS_AUTHORITY_POLICY_SCHEMA = "struct-xai-custody-key-status-authority-policy/v1"
CUSTODY_KEY_STATUS_PURPOSE = "publication-reference-custodian-key-status"
MAX_REFERENCE_CUSTODIANS = 16
MAX_CUSTODIED_AUDIT_RECEIPT_BYTES = 8 * 1024
MAX_CUSTODY_ATTESTATION_LIFETIME_SECONDS = 60 * 60
MAX_CUSTODY_CLOCK_SKEW_SECONDS = 5 * 60
MAX_CUSTODY_EPOCH = 253_402_300_799

_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_REFERENCE_FIELDS = {
    "schema_version",
    "publication_uri",
    "publication_record_sha256",
    "manifest_sha256",
    "published_at_epoch",
}
_APPROVAL_METADATA_FIELDS = {
    "custodian_id",
    "algorithm",
    "signature_encoding",
    "key_id",
    "key_revision_sha256",
}
_APPROVAL_FIELDS = _APPROVAL_METADATA_FIELDS | {"signature"}
_ATTESTATION_FIELDS = {
    "schema_version",
    "purpose",
    "publication_reference_sha256",
    "custodian_set_sha256",
    "required_approvals",
    "attested_at_epoch",
    "expires_at_epoch",
    "approvals",
    "attestation_sha256",
}
_RECEIPT_FIELDS = {
    "schema_version",
    "verified",
    "manifest_sha256",
    "publication_record_sha256",
    "publication_reference_sha256",
    "runtime_evidence_sha256",
    "base_audit_receipt_sha256",
    "reference_custody_attestation_sha256",
    "reference_custodian_set_sha256",
    "reference_custody_verification_sha256",
    "reference_required_approvals",
    "reference_valid_approvals",
    "receipt_sha256",
}
_RECEIPT_DIGEST_FIELDS = _RECEIPT_FIELDS - {
    "schema_version",
    "verified",
    "reference_required_approvals",
    "reference_valid_approvals",
}
_KEY_STATUS_ROW_FIELDS = {"key_id", "key_revision_sha256", "status"}
_KEY_STATUS_AUTHORITY_METADATA_FIELDS = {
    "authority_algorithm",
    "authority_key_id",
    "authority_key_revision_sha256",
    "signature_encoding",
}
_KEY_STATUS_AUTHORITY_SIGNATURE_FIELDS = _KEY_STATUS_AUTHORITY_METADATA_FIELDS | {"signature"}
_KEY_STATUS_SNAPSHOT_FIELDS = {
    "schema_version",
    "purpose",
    "issued_at_epoch",
    "expires_at_epoch",
    "required_authority_signatures",
    "authority_policy_sha256",
    "authority_signatures",
    "statuses",
    "snapshot_sha256",
}
_KEY_STATUS_VALUES = {"active", "revoked"}


class StabilityReferenceCustodyError(ValueError):
    """Stable failure from the publication-reference custody boundary."""

    code = "STABILITY_REFERENCE_CUSTODY_REJECTED"

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
        raise StabilityReferenceCustodyError("custody evidence is not canonical JSON") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _same(left: object, right: object) -> bool:
    return isinstance(left, str) and isinstance(right, str) and hmac.compare_digest(left, right)


def _require_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise StabilityReferenceCustodyError(f"{name} is invalid")
    return value


def _require_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise StabilityReferenceCustodyError(f"{name} is invalid")
    return value


def _require_epoch(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= MAX_CUSTODY_EPOCH:
        raise StabilityReferenceCustodyError(f"{name} is invalid")
    return value


def _normalize_reference(reference: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(reference, Mapping) or set(reference) != _REFERENCE_FIELDS:
        raise StabilityReferenceCustodyError("publication reference must use the exact schema")
    if reference.get("schema_version") != MANIFEST_PUBLICATION_REFERENCE_SCHEMA:
        raise StabilityReferenceCustodyError("publication reference metadata is invalid")
    try:
        return create_manifest_publication_reference(
            publication_uri=reference.get("publication_uri"),  # type: ignore[arg-type]
            publication_record_sha256=reference.get("publication_record_sha256"),  # type: ignore[arg-type]
            manifest_sha256=reference.get("manifest_sha256"),  # type: ignore[arg-type]
            published_at_epoch=reference.get("published_at_epoch"),  # type: ignore[arg-type]
        )
    except (StabilityLineageError, TypeError, ValueError) as exc:
        raise StabilityReferenceCustodyError("publication reference is invalid") from exc


def _adapter_metadata(
    adapter: ExternalEvidenceSigner | ExternalEvidenceVerifier,
) -> tuple[str, str, str]:
    try:
        algorithm = _require_identifier(adapter.algorithm, "algorithm")
        key_id = _require_identifier(adapter.key_id, "key_id")
        key_revision = _require_digest(adapter.key_revision_sha256, "key_revision_sha256")
    except StabilityReferenceCustodyError:
        raise
    except Exception as exc:
        raise StabilityReferenceCustodyError("external signature adapter metadata is invalid") from exc
    if algorithm not in EXTERNAL_SIGNATURE_ALGORITHMS:
        raise StabilityReferenceCustodyError("external signature algorithm is unsupported")
    return algorithm, key_id, key_revision


def _encode_signature(value: object) -> str:
    if (
        not isinstance(value, bytes)
        or not MIN_EXTERNAL_SIGNATURE_BYTES <= len(value) <= MAX_EXTERNAL_SIGNATURE_BYTES
    ):
        raise StabilityReferenceCustodyError("external signature is invalid")
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode_signature(value: object) -> bytes:
    if not isinstance(value, str) or not value or "=" in value:
        raise StabilityReferenceCustodyError("external signature is invalid")
    try:
        decoded = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
    except (binascii.Error, ValueError, UnicodeEncodeError) as exc:
        raise StabilityReferenceCustodyError("external signature is invalid") from exc
    if _encode_signature(decoded) != value:
        raise StabilityReferenceCustodyError("external signature is invalid")
    return decoded


def _approval_claims(
    approval: Mapping[str, object],
    *,
    publication_reference_sha256: str,
    custodian_set_sha256: str,
    required_approvals: int,
    attested_at_epoch: int,
    expires_at_epoch: int,
) -> dict[str, object]:
    return {
        "schema_version": PUBLICATION_REFERENCE_CUSTODY_SCHEMA,
        "purpose": PUBLICATION_REFERENCE_CUSTODY_PURPOSE,
        "publication_reference_sha256": publication_reference_sha256,
        "custodian_set_sha256": custodian_set_sha256,
        "required_approvals": required_approvals,
        "attested_at_epoch": attested_at_epoch,
        "expires_at_epoch": expires_at_epoch,
        **{field: approval[field] for field in _APPROVAL_METADATA_FIELDS},
    }


def _validate_approval_metadata(approval: Mapping[str, object]) -> dict[str, str]:
    if not isinstance(approval, Mapping) or set(approval) != _APPROVAL_FIELDS:
        raise StabilityReferenceCustodyError("each custody approval must use the exact schema")
    algorithm = _require_identifier(approval.get("algorithm"), "algorithm")
    if algorithm not in EXTERNAL_SIGNATURE_ALGORITHMS:
        raise StabilityReferenceCustodyError("external signature algorithm is unsupported")
    if approval.get("signature_encoding") != EXTERNAL_SIGNATURE_ENCODING:
        raise StabilityReferenceCustodyError("external signature encoding is invalid")
    _decode_signature(approval.get("signature"))
    return {
        "custodian_id": _require_identifier(approval.get("custodian_id"), "custodian_id"),
        "algorithm": algorithm,
        "signature_encoding": EXTERNAL_SIGNATURE_ENCODING,
        "key_id": _require_identifier(approval.get("key_id"), "key_id"),
        "key_revision_sha256": _require_digest(approval.get("key_revision_sha256"), "key_revision_sha256"),
    }


def _key_status_claims(snapshot: Mapping[str, object]) -> dict[str, object]:
    return {
        key: snapshot[key]
        for key in _KEY_STATUS_SNAPSHOT_FIELDS
        if key not in {"authority_signatures", "snapshot_sha256"}
    }


def _normalize_key_status_rows(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list) or not 2 <= len(value) <= MAX_REFERENCE_CUSTODIANS:
        raise StabilityReferenceCustodyError("key status rows are invalid")
    rows: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _KEY_STATUS_ROW_FIELDS:
            raise StabilityReferenceCustodyError("key status row must use the exact schema")
        status = item.get("status")
        if not isinstance(status, str) or status not in _KEY_STATUS_VALUES:
            raise StabilityReferenceCustodyError("custodian key status is invalid")
        rows.append(
            {
                "key_id": _require_identifier(item.get("key_id"), "key_id"),
                "key_revision_sha256": _require_digest(
                    item.get("key_revision_sha256"), "key_revision_sha256"
                ),
                "status": str(status),
            }
        )
    if rows != sorted(rows, key=lambda item: item["key_id"]):
        raise StabilityReferenceCustodyError("key status rows are not canonically ordered")
    if len({item["key_id"] for item in rows}) != len(rows):
        raise StabilityReferenceCustodyError("key status identifiers must be distinct")
    if len({item["key_revision_sha256"] for item in rows}) != len(rows):
        raise StabilityReferenceCustodyError("key status revisions must be distinct")
    return rows


def _key_status_authority_policy(
    authorities: Mapping[str, ExternalEvidenceSigner | ExternalEvidenceVerifier],
    *,
    required_signatures: int,
    operation: str,
) -> tuple[str, dict[str, tuple[str, str, str]]]:
    if not isinstance(authorities, Mapping) or not 2 <= len(authorities) <= MAX_REFERENCE_CUSTODIANS:
        raise StabilityReferenceCustodyError("key status authority set must contain from 2 through 16 keys")
    if type(required_signatures) is not int or not 2 <= required_signatures <= len(authorities):
        raise StabilityReferenceCustodyError("required key status authority signatures are invalid")
    metadata: dict[str, tuple[str, str, str]] = {}
    revisions: set[str] = set()
    for configured_key_id in sorted(authorities):
        adapter = authorities[configured_key_id]
        try:
            algorithm, key_id, revision = _adapter_metadata(adapter)
        except StabilityReferenceCustodyError as exc:
            raise StabilityReferenceCustodyError("key status authority metadata is invalid") from exc
        if configured_key_id != key_id or not callable(getattr(adapter, operation, None)):
            raise StabilityReferenceCustodyError("key status authority metadata mismatch")
        if revision in revisions:
            raise StabilityReferenceCustodyError("key status authority revisions must be distinct")
        revisions.add(revision)
        metadata[key_id] = (algorithm, key_id, revision)
    policy = {
        "schema_version": CUSTODY_KEY_STATUS_AUTHORITY_POLICY_SCHEMA,
        "required_signatures": required_signatures,
        "authorities": [
            {
                "algorithm": algorithm,
                "key_id": key_id,
                "key_revision_sha256": revision,
            }
            for algorithm, key_id, revision in metadata.values()
        ],
    }
    return _digest(policy), metadata


def _normalize_key_status_authority_signatures(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list) or not 2 <= len(value) <= MAX_REFERENCE_CUSTODIANS:
        raise StabilityReferenceCustodyError("key status authority signatures are invalid")
    signatures: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _KEY_STATUS_AUTHORITY_SIGNATURE_FIELDS:
            raise StabilityReferenceCustodyError("key status authority signature must use the exact schema")
        algorithm = _require_identifier(item.get("authority_algorithm"), "authority_algorithm")
        if algorithm not in EXTERNAL_SIGNATURE_ALGORITHMS:
            raise StabilityReferenceCustodyError("key status authority algorithm is unsupported")
        if item.get("signature_encoding") != EXTERNAL_SIGNATURE_ENCODING:
            raise StabilityReferenceCustodyError("key status authority signature encoding is invalid")
        signature = item.get("signature")
        _decode_signature(signature)
        signatures.append(
            {
                "authority_algorithm": algorithm,
                "authority_key_id": _require_identifier(item.get("authority_key_id"), "authority_key_id"),
                "authority_key_revision_sha256": _require_digest(
                    item.get("authority_key_revision_sha256"), "authority_key_revision_sha256"
                ),
                "signature_encoding": EXTERNAL_SIGNATURE_ENCODING,
                "signature": str(signature),
            }
        )
    if signatures != sorted(signatures, key=lambda item: item["authority_key_id"]):
        raise StabilityReferenceCustodyError("key status authority signatures are not canonically ordered")
    if len({item["authority_key_id"] for item in signatures}) != len(signatures):
        raise StabilityReferenceCustodyError("key status authority identifiers must be distinct")
    if len({item["authority_key_revision_sha256"] for item in signatures}) != len(signatures):
        raise StabilityReferenceCustodyError("key status authority revisions must be distinct")
    return signatures


def create_custody_key_status_snapshot(
    key_statuses: Mapping[str, tuple[str, str]],
    authorities: Mapping[str, ExternalEvidenceSigner],
    *,
    required_authority_signatures: int = 2,
    signing_authority_ids: tuple[str, ...] | None = None,
    issued_at_epoch: int,
    expires_at_epoch: int,
) -> dict[str, object]:
    """Threshold-sign one short-lived active/revoked view of the custodian keys."""

    if not isinstance(key_statuses, Mapping) or not 2 <= len(key_statuses) <= MAX_REFERENCE_CUSTODIANS:
        raise StabilityReferenceCustodyError("key_statuses must contain from 2 through 16 keys")
    issued_at = _require_epoch(issued_at_epoch, "issued_at_epoch")
    expires_at = _require_epoch(expires_at_epoch, "expires_at_epoch")
    if not issued_at < expires_at <= issued_at + MAX_CUSTODY_ATTESTATION_LIFETIME_SECONDS:
        raise StabilityReferenceCustodyError("key status validity window is invalid")
    rows: list[dict[str, str]] = []
    for raw_key_id, value in key_statuses.items():
        if not isinstance(value, tuple) or len(value) != 2:
            raise StabilityReferenceCustodyError("custodian key status is invalid")
        revision, status = value
        if not isinstance(status, str) or status not in _KEY_STATUS_VALUES:
            raise StabilityReferenceCustodyError("custodian key status is invalid")
        rows.append(
            {
                "key_id": _require_identifier(raw_key_id, "key_id"),
                "key_revision_sha256": _require_digest(revision, "key_revision_sha256"),
                "status": status,
            }
        )
    rows.sort(key=lambda item: item["key_id"])
    rows = _normalize_key_status_rows(rows)
    authority_policy_sha256, authority_metadata = _key_status_authority_policy(
        authorities,
        required_signatures=required_authority_signatures,
        operation="sign",
    )
    custodian_revisions = {item["key_revision_sha256"] for item in rows}
    if any(metadata[2] in custodian_revisions for metadata in authority_metadata.values()):
        raise StabilityReferenceCustodyError("key status authority must be independent")
    selected = tuple(authority_metadata) if signing_authority_ids is None else signing_authority_ids
    if (
        not isinstance(selected, tuple)
        or selected != tuple(sorted(set(selected)))
        or not required_authority_signatures <= len(selected) <= len(authority_metadata)
        or any(key_id not in authority_metadata for key_id in selected)
    ):
        raise StabilityReferenceCustodyError("key status signing authority set is invalid")
    claims: dict[str, object] = {
        "schema_version": CUSTODY_KEY_STATUS_SNAPSHOT_SCHEMA,
        "purpose": CUSTODY_KEY_STATUS_PURPOSE,
        "issued_at_epoch": issued_at,
        "expires_at_epoch": expires_at,
        "required_authority_signatures": required_authority_signatures,
        "authority_policy_sha256": authority_policy_sha256,
        "statuses": rows,
    }
    snapshot_sha256 = _digest(claims)
    message = _canonical_json({**claims, "snapshot_sha256": snapshot_sha256})
    authority_signatures: list[dict[str, str]] = []
    for key_id in selected:
        algorithm, _, revision = authority_metadata[key_id]
        try:
            signature = authorities[key_id].sign(message)
        except Exception as exc:
            raise StabilityReferenceCustodyError("key status signing failed") from exc
        authority_signatures.append(
            {
                "authority_algorithm": algorithm,
                "authority_key_id": key_id,
                "authority_key_revision_sha256": revision,
                "signature_encoding": EXTERNAL_SIGNATURE_ENCODING,
                "signature": _encode_signature(signature),
            }
        )
    return {
        **claims,
        "snapshot_sha256": snapshot_sha256,
        "authority_signatures": authority_signatures,
    }


def _verify_custody_key_status_snapshot(
    snapshot: Mapping[str, object],
    expected_keys: Mapping[str, str],
    authority_verifiers: Mapping[str, ExternalEvidenceVerifier],
    *,
    required_authority_signatures: int,
    verification_epoch: int,
    clock_skew_seconds: int,
) -> tuple[dict[str, object], frozenset[str]]:
    if not isinstance(snapshot, Mapping) or set(snapshot) != _KEY_STATUS_SNAPSHOT_FIELDS:
        raise StabilityReferenceCustodyError("key status snapshot must use the exact schema")
    document = dict(snapshot)
    if (
        document.get("schema_version") != CUSTODY_KEY_STATUS_SNAPSHOT_SCHEMA
        or document.get("purpose") != CUSTODY_KEY_STATUS_PURPOSE
    ):
        raise StabilityReferenceCustodyError("key status snapshot metadata is invalid")
    issued_at = _require_epoch(document.get("issued_at_epoch"), "issued_at_epoch")
    expires_at = _require_epoch(document.get("expires_at_epoch"), "expires_at_epoch")
    verified_at = _require_epoch(verification_epoch, "verification_epoch")
    if type(clock_skew_seconds) is not int or not 0 <= clock_skew_seconds <= MAX_CUSTODY_CLOCK_SKEW_SECONDS:
        raise StabilityReferenceCustodyError("clock_skew_seconds is invalid")
    if not issued_at < expires_at <= issued_at + MAX_CUSTODY_ATTESTATION_LIFETIME_SECONDS:
        raise StabilityReferenceCustodyError("key status validity window is invalid")
    if verified_at < issued_at - clock_skew_seconds:
        raise StabilityReferenceCustodyError("key status snapshot is from the future")
    if verified_at > expires_at:
        raise StabilityReferenceCustodyError("key status snapshot has expired")
    rows = _normalize_key_status_rows(document.get("statuses"))
    if not isinstance(expected_keys, Mapping) or not 2 <= len(expected_keys) <= MAX_REFERENCE_CUSTODIANS:
        raise StabilityReferenceCustodyError("expected custodian keys are invalid")
    normalized_expected = {
        _require_identifier(key_id, "key_id"): _require_digest(revision, "key_revision_sha256")
        for key_id, revision in expected_keys.items()
    }
    observed = {item["key_id"]: item["key_revision_sha256"] for item in rows}
    if observed != normalized_expected:
        raise StabilityReferenceCustodyError("key status snapshot does not cover the custodian set")
    claimed_required = document.get("required_authority_signatures")
    if claimed_required != required_authority_signatures:
        raise StabilityReferenceCustodyError("key status authority threshold mismatch")
    expected_policy_sha256, trusted_authorities = _key_status_authority_policy(
        authority_verifiers,
        required_signatures=required_authority_signatures,
        operation="verify",
    )
    if not _same(document.get("authority_policy_sha256"), expected_policy_sha256):
        raise StabilityReferenceCustodyError("key status authority policy mismatch")
    if any(metadata[2] in set(observed.values()) for metadata in trusted_authorities.values()):
        raise StabilityReferenceCustodyError("key status authority must be independent")
    claimed_snapshot_sha256 = document.get("snapshot_sha256")
    if not _same(claimed_snapshot_sha256, _digest(_key_status_claims(document))):
        raise StabilityReferenceCustodyError("key status snapshot digest does not match its content")
    authority_signatures = _normalize_key_status_authority_signatures(document.get("authority_signatures"))
    if len(authority_signatures) < required_authority_signatures:
        raise StabilityReferenceCustodyError("key status authority quorum is not satisfied")
    message = _canonical_json({**_key_status_claims(document), "snapshot_sha256": claimed_snapshot_sha256})
    valid_authority_revisions: set[str] = set()
    for item in authority_signatures:
        key_id = item["authority_key_id"]
        trusted_metadata = trusted_authorities.get(key_id)
        observed_metadata = (
            item["authority_algorithm"],
            key_id,
            item["authority_key_revision_sha256"],
        )
        verifier = authority_verifiers.get(key_id)
        if trusted_metadata != observed_metadata or verifier is None:
            raise StabilityReferenceCustodyError("key status authority metadata mismatch")
        try:
            verified = verifier.verify(message, _decode_signature(item["signature"]))
        except Exception:  # noqa: BLE001 - provider failures are stable verification failures
            verified = False
        if verified is not True:
            raise StabilityReferenceCustodyError("key status authority signature is invalid")
        valid_authority_revisions.add(item["authority_key_revision_sha256"])
    if len(valid_authority_revisions) < required_authority_signatures:
        raise StabilityReferenceCustodyError("key status authority quorum is not satisfied")
    active_revisions = frozenset(item["key_revision_sha256"] for item in rows if item["status"] == "active")
    verification: dict[str, object] = {
        "schema_version": CUSTODY_KEY_STATUS_VERIFICATION_SCHEMA,
        "verified": True,
        "snapshot_sha256": claimed_snapshot_sha256,
        "verified_at_epoch": verified_at,
        "authority_policy_sha256": expected_policy_sha256,
        "required_authority_signatures": required_authority_signatures,
        "valid_authority_signatures": len(valid_authority_revisions),
        "active_key_count": len(active_revisions),
        "revoked_key_count": len(rows) - len(active_revisions),
        "active_key_revisions_sha256": _digest(sorted(active_revisions)),
    }
    verification["verification_sha256"] = _digest(verification)
    return verification, active_revisions


def verify_custody_key_status_snapshot(
    snapshot: Mapping[str, object],
    expected_keys: Mapping[str, str],
    authority_verifiers: Mapping[str, ExternalEvidenceVerifier],
    *,
    required_authority_signatures: int = 2,
    verification_epoch: int,
    clock_skew_seconds: int = MAX_CUSTODY_CLOCK_SKEW_SECONDS,
) -> dict[str, object]:
    """Verify one exact short-lived custodian key-status view."""

    verification, _ = _verify_custody_key_status_snapshot(
        snapshot,
        expected_keys,
        authority_verifiers,
        required_authority_signatures=required_authority_signatures,
        verification_epoch=verification_epoch,
        clock_skew_seconds=clock_skew_seconds,
    )
    return verification


def create_publication_reference_custody_attestation(
    publication_reference: Mapping[str, object],
    custodians: Mapping[str, ExternalEvidenceSigner],
    *,
    required_approvals: int = 2,
    attested_at_epoch: int,
    expires_at_epoch: int,
) -> dict[str, object]:
    """Create a short-lived quorum attestation over one exact publication reference."""

    reference = _normalize_reference(publication_reference)
    if not isinstance(custodians, Mapping) or not 2 <= len(custodians) <= MAX_REFERENCE_CUSTODIANS:
        raise StabilityReferenceCustodyError("custodians must contain from 2 through 16 signers")
    if (
        isinstance(required_approvals, bool)
        or not isinstance(required_approvals, int)
        or not 2 <= required_approvals <= len(custodians)
    ):
        raise StabilityReferenceCustodyError("required_approvals is invalid")
    attested_at = _require_epoch(attested_at_epoch, "attested_at_epoch")
    expires_at = _require_epoch(expires_at_epoch, "expires_at_epoch")
    if (
        attested_at < int(reference["published_at_epoch"])
        or not attested_at < expires_at <= attested_at + MAX_CUSTODY_ATTESTATION_LIFETIME_SECONDS
    ):
        raise StabilityReferenceCustodyError("custody attestation validity window is invalid")

    signer_by_custodian: dict[str, ExternalEvidenceSigner] = {}
    metadata: list[dict[str, str]] = []
    for raw_custodian_id, signer in custodians.items():
        custodian_id = _require_identifier(raw_custodian_id, "custodian_id")
        algorithm, key_id, key_revision = _adapter_metadata(signer)
        if not callable(getattr(signer, "sign", None)):
            raise StabilityReferenceCustodyError("external signer is invalid")
        signer_by_custodian[custodian_id] = signer
        metadata.append(
            {
                "custodian_id": custodian_id,
                "algorithm": algorithm,
                "signature_encoding": EXTERNAL_SIGNATURE_ENCODING,
                "key_id": key_id,
                "key_revision_sha256": key_revision,
            }
        )
    metadata.sort(key=lambda value: value["custodian_id"])
    if len({item["custodian_id"] for item in metadata}) != len(metadata):
        raise StabilityReferenceCustodyError("custodian identifiers must be distinct")
    if len({item["key_id"] for item in metadata}) != len(metadata):
        raise StabilityReferenceCustodyError("custodian key identifiers must be distinct")
    if len({item["key_revision_sha256"] for item in metadata}) != len(metadata):
        raise StabilityReferenceCustodyError("custodian key revisions must be distinct")

    reference_sha256 = _digest(reference)
    custodian_set_sha256 = _digest(metadata)
    approvals: list[dict[str, object]] = []
    for item in metadata:
        claims = _approval_claims(
            item,
            publication_reference_sha256=reference_sha256,
            custodian_set_sha256=custodian_set_sha256,
            required_approvals=required_approvals,
            attested_at_epoch=attested_at,
            expires_at_epoch=expires_at,
        )
        try:
            signature = signer_by_custodian[item["custodian_id"]].sign(_canonical_json(claims))
        except Exception as exc:
            raise StabilityReferenceCustodyError("external custody signing failed") from exc
        approvals.append({**item, "signature": _encode_signature(signature)})

    attestation: dict[str, object] = {
        "schema_version": PUBLICATION_REFERENCE_CUSTODY_SCHEMA,
        "purpose": PUBLICATION_REFERENCE_CUSTODY_PURPOSE,
        "publication_reference_sha256": reference_sha256,
        "custodian_set_sha256": custodian_set_sha256,
        "required_approvals": required_approvals,
        "attested_at_epoch": attested_at,
        "expires_at_epoch": expires_at,
        "approvals": approvals,
    }
    attestation["attestation_sha256"] = _digest(attestation)
    return attestation


def verify_publication_reference_custody_attestation(
    publication_reference: Mapping[str, object],
    attestation: Mapping[str, object],
    verifiers: Mapping[str, ExternalEvidenceVerifier],
    *,
    verification_epoch: int,
    clock_skew_seconds: int = MAX_CUSTODY_CLOCK_SKEW_SECONDS,
) -> dict[str, object]:
    """Require a fresh threshold of independently configured custody keys."""

    reference = _normalize_reference(publication_reference)
    if not isinstance(attestation, Mapping) or set(attestation) != _ATTESTATION_FIELDS:
        raise StabilityReferenceCustodyError("custody attestation must use the exact schema")
    document = dict(attestation)
    if (
        document.get("schema_version") != PUBLICATION_REFERENCE_CUSTODY_SCHEMA
        or document.get("purpose") != PUBLICATION_REFERENCE_CUSTODY_PURPOSE
    ):
        raise StabilityReferenceCustodyError("custody attestation metadata is invalid")
    required = document.get("required_approvals")
    approvals = document.get("approvals")
    if (
        isinstance(required, bool)
        or not isinstance(required, int)
        or not isinstance(approvals, list)
        or not 2 <= required <= len(approvals) <= MAX_REFERENCE_CUSTODIANS
    ):
        raise StabilityReferenceCustodyError("custody approval threshold is invalid")
    if not isinstance(verifiers, Mapping):
        raise StabilityReferenceCustodyError("verifiers must be a mapping")
    attested_at = _require_epoch(document.get("attested_at_epoch"), "attested_at_epoch")
    expires_at = _require_epoch(document.get("expires_at_epoch"), "expires_at_epoch")
    verified_at = _require_epoch(verification_epoch, "verification_epoch")
    if type(clock_skew_seconds) is not int or not 0 <= clock_skew_seconds <= MAX_CUSTODY_CLOCK_SKEW_SECONDS:
        raise StabilityReferenceCustodyError("clock_skew_seconds is invalid")
    if (
        attested_at < int(reference["published_at_epoch"])
        or not attested_at < expires_at <= attested_at + MAX_CUSTODY_ATTESTATION_LIFETIME_SECONDS
    ):
        raise StabilityReferenceCustodyError("custody attestation validity window is invalid")
    if verified_at < attested_at - clock_skew_seconds:
        raise StabilityReferenceCustodyError("custody attestation is from the future")
    if verified_at > expires_at:
        raise StabilityReferenceCustodyError("custody attestation has expired")

    normalized_metadata: list[dict[str, str]] = []
    for approval in approvals:
        normalized_metadata.append(_validate_approval_metadata(approval))
    if normalized_metadata != sorted(normalized_metadata, key=lambda value: value["custodian_id"]):
        raise StabilityReferenceCustodyError("custody approvals are not canonically ordered")
    if len({item["custodian_id"] for item in normalized_metadata}) != len(approvals):
        raise StabilityReferenceCustodyError("custodian identifiers must be distinct")
    if len({item["key_id"] for item in normalized_metadata}) != len(approvals):
        raise StabilityReferenceCustodyError("custodian key identifiers must be distinct")
    if len({item["key_revision_sha256"] for item in normalized_metadata}) != len(approvals):
        raise StabilityReferenceCustodyError("custodian key revisions must be distinct")

    reference_sha256 = _digest(reference)
    custodian_set_sha256 = _digest(normalized_metadata)
    if not _same(document.get("publication_reference_sha256"), reference_sha256):
        raise StabilityReferenceCustodyError("custody attestation names a different publication reference")
    if not _same(document.get("custodian_set_sha256"), custodian_set_sha256):
        raise StabilityReferenceCustodyError("custodian set digest does not match its content")
    claimed_attestation_sha256 = document.pop("attestation_sha256")
    if not _same(claimed_attestation_sha256, _digest(document)):
        raise StabilityReferenceCustodyError("custody attestation digest does not match its content")

    valid_revisions: list[str] = []
    for approval, metadata in zip(approvals, normalized_metadata, strict=True):
        verifier = verifiers.get(metadata["key_id"])
        if verifier is None:
            continue
        try:
            verifier_metadata = _adapter_metadata(verifier)
        except StabilityReferenceCustodyError:
            continue
        if verifier_metadata != (
            metadata["algorithm"],
            metadata["key_id"],
            metadata["key_revision_sha256"],
        ) or not callable(getattr(verifier, "verify", None)):
            continue
        claims = _approval_claims(
            metadata,
            publication_reference_sha256=reference_sha256,
            custodian_set_sha256=custodian_set_sha256,
            required_approvals=required,
            attested_at_epoch=attested_at,
            expires_at_epoch=expires_at,
        )
        try:
            verified = verifier.verify(
                _canonical_json(claims),
                _decode_signature(approval.get("signature")),
            )
        except Exception:  # noqa: BLE001 - provider failures count as invalid approvals
            verified = False
        if verified is True:
            valid_revisions.append(metadata["key_revision_sha256"])
    if len(valid_revisions) < required:
        raise StabilityReferenceCustodyError("custody quorum is not satisfied")

    verification: dict[str, object] = {
        "schema_version": PUBLICATION_REFERENCE_CUSTODY_VERIFICATION_SCHEMA,
        "verified": True,
        "publication_reference_sha256": reference_sha256,
        "custodian_set_sha256": custodian_set_sha256,
        "attestation_sha256": claimed_attestation_sha256,
        "required_approvals": required,
        "valid_approvals": len(valid_revisions),
        "attested_at_epoch": attested_at,
        "expires_at_epoch": expires_at,
        "verified_at_epoch": verified_at,
        "clock_skew_seconds": clock_skew_seconds,
        "valid_key_revisions_sha256": _digest(sorted(valid_revisions)),
    }
    verification["verification_sha256"] = _digest(verification)
    return verification


def verify_publication_reference_custody_with_key_status(
    publication_reference: Mapping[str, object],
    attestation: Mapping[str, object],
    verifiers: Mapping[str, ExternalEvidenceVerifier],
    key_status_snapshot: Mapping[str, object],
    key_status_authority_verifiers: Mapping[str, ExternalEvidenceVerifier],
    *,
    required_key_status_authority_signatures: int = 2,
    verification_epoch: int,
    clock_skew_seconds: int = MAX_CUSTODY_CLOCK_SKEW_SECONDS,
) -> dict[str, object]:
    """Exclude revoked custodian revisions before calculating the custody quorum."""

    if not isinstance(attestation, Mapping) or set(attestation) != _ATTESTATION_FIELDS:
        raise StabilityReferenceCustodyError("custody attestation must use the exact schema")
    approvals = attestation.get("approvals")
    if not isinstance(approvals, list):
        raise StabilityReferenceCustodyError("custody approval threshold is invalid")
    if not isinstance(verifiers, Mapping):
        raise StabilityReferenceCustodyError("verifiers must be a mapping")
    metadata = [_validate_approval_metadata(approval) for approval in approvals]
    expected_keys = {item["key_id"]: item["key_revision_sha256"] for item in metadata}
    key_status, active_revisions = _verify_custody_key_status_snapshot(
        key_status_snapshot,
        expected_keys,
        key_status_authority_verifiers,
        required_authority_signatures=required_key_status_authority_signatures,
        verification_epoch=verification_epoch,
        clock_skew_seconds=clock_skew_seconds,
    )
    active_verifiers: dict[str, ExternalEvidenceVerifier] = {}
    for key_id, verifier in verifiers.items():
        try:
            _algorithm, verifier_key_id, revision = _adapter_metadata(verifier)
        except StabilityReferenceCustodyError:
            continue
        if key_id == verifier_key_id and revision in active_revisions:
            active_verifiers[key_id] = verifier
    custody = verify_publication_reference_custody_attestation(
        publication_reference,
        attestation,
        active_verifiers,
        verification_epoch=verification_epoch,
        clock_skew_seconds=clock_skew_seconds,
    )
    verification: dict[str, object] = {
        key: value for key, value in custody.items() if key not in {"schema_version", "verification_sha256"}
    }
    verification.update(
        {
            "schema_version": PUBLICATION_REFERENCE_CUSTODY_REVOCATION_VERIFICATION_SCHEMA,
            "base_custody_verification_sha256": custody["verification_sha256"],
            "key_status_snapshot_sha256": key_status["snapshot_sha256"],
            "key_status_verification_sha256": key_status["verification_sha256"],
            "active_key_count": key_status["active_key_count"],
            "revoked_key_count": key_status["revoked_key_count"],
        }
    )
    verification["verification_sha256"] = _digest(verification)
    return verification


def verify_custodied_stability_audit_bundle(
    publication_reference: Mapping[str, object],
    custody_attestation: Mapping[str, object],
    publication_record_bytes: bytes,
    *,
    retrieved_uri: str,
    lineage_bundle: Mapping[str, object],
    custody_verifiers: Mapping[str, ExternalEvidenceVerifier],
    lineage_verifiers: Mapping[str, ExternalEvidenceVerifier],
    verification_epoch: int,
    clock_skew_seconds: int = MAX_CUSTODY_CLOCK_SKEW_SECONDS,
    key_status_snapshot: Mapping[str, object] | None = None,
    key_status_authority_verifiers: Mapping[str, ExternalEvidenceVerifier] | None = None,
    required_key_status_authority_signatures: int = 2,
) -> dict[str, object]:
    """Verify reference custody, publication bytes, and signed runtime lineage as one unit."""

    try:
        if key_status_snapshot is None and key_status_authority_verifiers is None:
            custody = verify_publication_reference_custody_attestation(
                publication_reference,
                custody_attestation,
                custody_verifiers,
                verification_epoch=verification_epoch,
                clock_skew_seconds=clock_skew_seconds,
            )
        elif key_status_snapshot is not None and key_status_authority_verifiers is not None:
            custody = verify_publication_reference_custody_with_key_status(
                publication_reference,
                custody_attestation,
                custody_verifiers,
                key_status_snapshot,
                key_status_authority_verifiers,
                required_key_status_authority_signatures=required_key_status_authority_signatures,
                verification_epoch=verification_epoch,
                clock_skew_seconds=clock_skew_seconds,
            )
        else:
            raise StabilityReferenceCustodyError("key status inputs must be supplied together")
    except (StabilityReferenceCustodyError, TypeError, ValueError) as exc:
        raise StabilityReferenceCustodyError("reference custody verification failed") from exc
    try:
        audit = verify_stability_audit_bundle(
            publication_reference,
            publication_record_bytes,
            retrieved_uri=retrieved_uri,
            lineage_bundle=lineage_bundle,
            verifiers=lineage_verifiers,
        )
    except (StabilityAuditBundleError, TypeError, ValueError) as exc:
        raise StabilityReferenceCustodyError("stability audit verification failed") from exc

    receipt: dict[str, object] = {
        "schema_version": CUSTODIED_STABILITY_AUDIT_RECEIPT_SCHEMA,
        "verified": True,
        "manifest_sha256": audit["manifest_sha256"],
        "publication_record_sha256": audit["publication_record_sha256"],
        "publication_reference_sha256": custody["publication_reference_sha256"],
        "runtime_evidence_sha256": audit["runtime_evidence_sha256"],
        "base_audit_receipt_sha256": audit["receipt_sha256"],
        "reference_custody_attestation_sha256": custody["attestation_sha256"],
        "reference_custodian_set_sha256": custody["custodian_set_sha256"],
        "reference_custody_verification_sha256": custody["verification_sha256"],
        "reference_required_approvals": custody["required_approvals"],
        "reference_valid_approvals": custody["valid_approvals"],
    }
    receipt["receipt_sha256"] = _digest(receipt)
    return receipt


def serialize_custodied_stability_audit_receipt(receipt: Mapping[str, object]) -> bytes:
    """Serialize a strict privacy-reduced receipt after all three checks pass."""

    if not isinstance(receipt, Mapping) or set(receipt) != _RECEIPT_FIELDS:
        raise StabilityReferenceCustodyError("custodied audit receipt must use the exact schema")
    document = dict(receipt)
    required = document.get("reference_required_approvals")
    valid = document.get("reference_valid_approvals")
    if (
        document.get("schema_version") != CUSTODIED_STABILITY_AUDIT_RECEIPT_SCHEMA
        or document.get("verified") is not True
        or isinstance(required, bool)
        or not isinstance(required, int)
        or isinstance(valid, bool)
        or not isinstance(valid, int)
        or not 2 <= required <= valid <= MAX_REFERENCE_CUSTODIANS
    ):
        raise StabilityReferenceCustodyError("custodied audit receipt metadata is invalid")
    for field in _RECEIPT_DIGEST_FIELDS:
        _require_digest(document.get(field), field)
    claimed = document.pop("receipt_sha256")
    if not _same(claimed, _digest(document)):
        raise StabilityReferenceCustodyError("custodied audit receipt digest does not match its content")
    document["receipt_sha256"] = claimed
    payload = _canonical_json(document)
    if len(payload) > MAX_CUSTODIED_AUDIT_RECEIPT_BYTES:
        raise StabilityReferenceCustodyError("custodied audit receipt exceeds the size limit")
    return payload
