from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
from collections.abc import Mapping
from typing import Any

from structxai.stability_manifest import (
    EXTERNAL_SIGNATURE_ALGORITHMS,
    EXTERNAL_SIGNATURE_ENCODING,
    MAX_EXTERNAL_SIGNATURE_BYTES,
    MIN_EXTERNAL_SIGNATURE_BYTES,
    ExternalEvidenceSigner,
    ExternalEvidenceVerifier,
)
from structxai.stability_status_transparency import (
    CustodyStatusTransparencyError,
    validate_custody_status_checkpoint,
    verify_custody_status_transparency_log,
)

SIGNED_STATUS_CHECKPOINT_SCHEMA = "struct-xai-signed-custody-status-checkpoint/v1"
SIGNED_STATUS_CHECKPOINT_PURPOSE = "custody-status-transparency-checkpoint"
SIGNED_STATUS_CHECKPOINT_VERIFICATION_SCHEMA = "struct-xai-signed-custody-status-checkpoint-verification/v1"
MAX_SIGNED_STATUS_CHECKPOINT_BYTES = 32 * 1024
MAX_SIGNED_STATUS_CHECKPOINT_EPOCH = 253_402_300_799

_DOMAIN = b"struct-xai/signed-custody-status-checkpoint/v1\0"
_SIGNER_POLICY_DOMAIN = b"struct-xai/custody-status-checkpoint-signer-policy/v1\0"
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_REVISION_RE = re.compile(r"[0-9a-f]{40,64}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_FIELDS = {
    "schema_version",
    "purpose",
    "checkpoint",
    "source_revision",
    "signed_at_epoch",
    "algorithm",
    "signature_encoding",
    "key_id",
    "key_revision_sha256",
    "signature",
    "envelope_sha256",
}
_METADATA_FIELDS = {
    "algorithm",
    "signature_encoding",
    "key_id",
    "key_revision_sha256",
}


class SignedCustodyStatusCheckpointError(ValueError):
    """Stable failure from the signed status-checkpoint boundary."""

    code = "SIGNED_CUSTODY_STATUS_CHECKPOINT_REJECTED"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


def create_signed_custody_status_checkpoint(
    checkpoint: Mapping[str, object],
    signer: ExternalEvidenceSigner,
    *,
    log_id: str,
    source_revision: str,
    signed_at_epoch: int,
) -> dict[str, object]:
    """Sign one exact checkpoint with an immutable external key revision."""

    normalized = _checkpoint(checkpoint, log_id=log_id)
    metadata = _adapter_metadata(signer)
    claims: dict[str, object] = {
        "schema_version": SIGNED_STATUS_CHECKPOINT_SCHEMA,
        "purpose": SIGNED_STATUS_CHECKPOINT_PURPOSE,
        "checkpoint": normalized,
        "source_revision": _source_revision(source_revision),
        "signed_at_epoch": _epoch(signed_at_epoch),
        **metadata,
    }
    try:
        signature = signer.sign(_DOMAIN + _canonical_json(claims))
    except Exception as exc:
        raise SignedCustodyStatusCheckpointError("checkpoint signing failed") from exc
    envelope = {
        **claims,
        "signature": _encode_signature(signature),
    }
    envelope["envelope_sha256"] = _digest(envelope)
    return envelope


def verify_signed_custody_status_checkpoint(
    envelope: Mapping[str, object],
    verifier: ExternalEvidenceVerifier,
    log_path: str | os.PathLike[str],
    *,
    expected_log_id: str,
    expected_source_revision: str,
) -> dict[str, object]:
    """Authenticate a retained checkpoint and prove the current log extends it."""

    normalized = _validate_envelope(envelope)
    checkpoint = _checkpoint(normalized["checkpoint"], log_id=expected_log_id)  # type: ignore[arg-type]
    expected_revision = _source_revision(expected_source_revision)
    metadata = _adapter_metadata(verifier)
    if any(normalized[field] != metadata[field] for field in _METADATA_FIELDS):
        raise SignedCustodyStatusCheckpointError("checkpoint verifier metadata mismatch")
    claims = {key: normalized[key] for key in _FIELDS if key not in {"signature", "envelope_sha256"}}
    try:
        verified = verifier.verify(
            _DOMAIN + _canonical_json(claims),
            _decode_signature(normalized["signature"]),
        )
    except Exception as exc:
        raise SignedCustodyStatusCheckpointError("checkpoint signature verification failed") from exc
    if verified is not True:
        raise SignedCustodyStatusCheckpointError("checkpoint signature is invalid")
    if not hmac.compare_digest(str(normalized["source_revision"]), expected_revision):
        raise SignedCustodyStatusCheckpointError("checkpoint source revision mismatch")
    try:
        current = verify_custody_status_transparency_log(
            log_path,
            log_id=expected_log_id,
            prior_checkpoint=checkpoint,
        )
    except (CustodyStatusTransparencyError, OSError, TypeError, ValueError) as exc:
        raise SignedCustodyStatusCheckpointError(
            "current transparency log does not extend the signed checkpoint"
        ) from exc
    signer_policy_sha256 = hashlib.sha256(_SIGNER_POLICY_DOMAIN + _canonical_json(metadata)).hexdigest()
    receipt: dict[str, object] = {
        "schema_version": SIGNED_STATUS_CHECKPOINT_VERIFICATION_SCHEMA,
        "verified": True,
        "signed_checkpoint_sha256": normalized["envelope_sha256"],
        "retained_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "current_checkpoint_sha256": current["checkpoint_sha256"],
        "retained_tree_size": checkpoint["tree_size"],
        "current_tree_size": current["tree_size"],
        "source_revision": expected_revision,
        "signer_policy_sha256": signer_policy_sha256,
    }
    receipt["verification_sha256"] = _digest(receipt)
    return receipt


def serialize_signed_custody_status_checkpoint(
    envelope: Mapping[str, object],
) -> bytes:
    """Serialize one validated signed checkpoint as canonical JSON."""

    return _canonical_json(_validate_envelope(envelope))


def load_signed_custody_status_checkpoint(payload: bytes) -> dict[str, object]:
    """Load one bounded, exact-schema, canonical signed checkpoint."""

    if not isinstance(payload, bytes) or not 1 <= len(payload) <= MAX_SIGNED_STATUS_CHECKPOINT_BYTES:
        raise SignedCustodyStatusCheckpointError("signed checkpoint payload is invalid")
    try:
        value = json.loads(payload, object_pairs_hook=_reject_duplicate_fields)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
        raise SignedCustodyStatusCheckpointError("signed checkpoint JSON is invalid") from exc
    normalized = _validate_envelope(value)
    if payload != _canonical_json(normalized):
        raise SignedCustodyStatusCheckpointError("signed checkpoint JSON is not canonical")
    return normalized


def _validate_envelope(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        raise SignedCustodyStatusCheckpointError("signed checkpoint must use the exact schema")
    envelope = dict(value)
    if (
        envelope.get("schema_version") != SIGNED_STATUS_CHECKPOINT_SCHEMA
        or envelope.get("purpose") != SIGNED_STATUS_CHECKPOINT_PURPOSE
        or envelope.get("signature_encoding") != EXTERNAL_SIGNATURE_ENCODING
    ):
        raise SignedCustodyStatusCheckpointError("signed checkpoint metadata is invalid")
    _identifier(envelope.get("algorithm"), "algorithm")
    if envelope["algorithm"] not in EXTERNAL_SIGNATURE_ALGORITHMS:
        raise SignedCustodyStatusCheckpointError("signature algorithm is unsupported")
    _identifier(envelope.get("key_id"), "key_id")
    _require_digest(envelope.get("key_revision_sha256"), "key_revision_sha256")
    _source_revision(envelope.get("source_revision"))
    _epoch(envelope.get("signed_at_epoch"))
    _decode_signature(envelope.get("signature"))
    checkpoint = envelope.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise SignedCustodyStatusCheckpointError("checkpoint is invalid")
    try:
        normalized_checkpoint = validate_custody_status_checkpoint(
            checkpoint,
            log_id=str(checkpoint.get("log_id")),
        )
    except (CustodyStatusTransparencyError, TypeError, ValueError) as exc:
        raise SignedCustodyStatusCheckpointError("checkpoint is invalid") from exc
    envelope["checkpoint"] = normalized_checkpoint
    claimed = _require_digest(envelope.get("envelope_sha256"), "envelope_sha256")
    unsigned_digest = _digest({key: envelope[key] for key in _FIELDS if key != "envelope_sha256"})
    if not hmac.compare_digest(claimed, unsigned_digest):
        raise SignedCustodyStatusCheckpointError("signed checkpoint digest does not match")
    return envelope


def _checkpoint(value: Mapping[str, object], *, log_id: str) -> dict[str, object]:
    try:
        return validate_custody_status_checkpoint(value, log_id=log_id)
    except (CustodyStatusTransparencyError, TypeError, ValueError) as exc:
        raise SignedCustodyStatusCheckpointError("checkpoint is invalid") from exc


def _adapter_metadata(
    adapter: ExternalEvidenceSigner | ExternalEvidenceVerifier,
) -> dict[str, str]:
    try:
        algorithm = _identifier(adapter.algorithm, "algorithm")
        key_id = _identifier(adapter.key_id, "key_id")
        key_revision = _require_digest(
            adapter.key_revision_sha256,
            "key_revision_sha256",
        )
    except SignedCustodyStatusCheckpointError:
        raise
    except Exception as exc:
        raise SignedCustodyStatusCheckpointError("external signature adapter metadata is invalid") from exc
    if algorithm not in EXTERNAL_SIGNATURE_ALGORITHMS:
        raise SignedCustodyStatusCheckpointError("signature algorithm is unsupported")
    return {
        "algorithm": algorithm,
        "signature_encoding": EXTERNAL_SIGNATURE_ENCODING,
        "key_id": key_id,
        "key_revision_sha256": key_revision,
    }


def _encode_signature(value: object) -> str:
    if (
        not isinstance(value, bytes)
        or not MIN_EXTERNAL_SIGNATURE_BYTES <= len(value) <= MAX_EXTERNAL_SIGNATURE_BYTES
    ):
        raise SignedCustodyStatusCheckpointError("external signature is invalid")
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode_signature(value: object) -> bytes:
    if not isinstance(value, str) or not value or "=" in value:
        raise SignedCustodyStatusCheckpointError("external signature is invalid")
    try:
        decoded = base64.b64decode(
            value + "=" * (-len(value) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, ValueError, UnicodeEncodeError) as exc:
        raise SignedCustodyStatusCheckpointError("external signature is invalid") from exc
    if _encode_signature(decoded) != value:
        raise SignedCustodyStatusCheckpointError("external signature is invalid")
    return decoded


def _source_revision(value: object) -> str:
    if not isinstance(value, str) or _REVISION_RE.fullmatch(value) is None:
        raise SignedCustodyStatusCheckpointError("source revision is invalid")
    return value


def _epoch(value: object) -> int:
    if type(value) is not int or not 0 <= value <= MAX_SIGNED_STATUS_CHECKPOINT_EPOCH:
        raise SignedCustodyStatusCheckpointError("checkpoint signing epoch is invalid")
    return value


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise SignedCustodyStatusCheckpointError(f"{name} is invalid")
    return value


def _require_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise SignedCustodyStatusCheckpointError(f"{name} is invalid")
    return value


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


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
        raise SignedCustodyStatusCheckpointError("signed checkpoint is not canonical JSON") from exc


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SignedCustodyStatusCheckpointError("signed checkpoint contains a duplicate field")
        result[key] = value
    return result
