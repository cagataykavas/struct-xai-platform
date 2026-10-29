from __future__ import annotations

import hashlib
import hmac
import json
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from structxai.patch_run_receipt import PatchRunReceiptError
from structxai.rank_ledger_backup import (
    MAX_MANIFEST_BYTES,
    restore_rank_ledger_backup,
    verify_rank_ledger_backup,
)

HANDOFF_SCHEMA = "struct-xai-rank-ledger-handoff/v1"
HANDOFF_VERIFICATION_SCHEMA = "struct-xai-rank-ledger-handoff-verification/v1"
AUTHORIZED_RESTORE_SCHEMA = "struct-xai-authorized-rank-ledger-restore/v1"
HANDOFF_ALGORITHM = "HMAC-SHA256"
HANDOFF_PURPOSE = "rank-ledger-standby-activation"
MAX_AUTHORITY_EPOCH = 2**63 - 1
MAX_HANDOFF_LIFETIME_SECONDS = 900
MAX_CLOCK_SKEW_SECONDS = 5
MINIMUM_KEY_BYTES = 32
MAX_HANDOFF_BYTES = MAX_MANIFEST_BYTES
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HANDOFF_FIELDS = {
    "schema_version",
    "algorithm",
    "purpose",
    "key_id",
    "authority_epoch",
    "source_controller_id",
    "standby_id",
    "issued_at_epoch",
    "expires_at_epoch",
    "backup_manifest_sha256",
    "backup_sha256",
    "promotion_manifest_sha256",
    "downstream_fence_sha256",
    "signature",
}


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
        raise PatchRunReceiptError("LEDGER_HANDOFF_NOT_CANONICAL") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _claims(handoff: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in handoff.items() if key != "signature"}


def _validate_identifier(value: object, code: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise PatchRunReceiptError(code)
    return value


def _validate_digest(value: object, code: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PatchRunReceiptError(code)
    return value


def _validate_key(key: object) -> bytes:
    if not isinstance(key, bytes) or len(key) < MINIMUM_KEY_BYTES:
        raise PatchRunReceiptError("LEDGER_HANDOFF_KEY_INVALID")
    return key


def _validate_epoch(value: object) -> int:
    if type(value) is not int or not 1 <= value <= MAX_AUTHORITY_EPOCH:
        raise PatchRunReceiptError("LEDGER_HANDOFF_EPOCH_INVALID")
    return value


def _validate_timestamp(value: object) -> int:
    if type(value) is not int or not 0 <= value <= MAX_AUTHORITY_EPOCH:
        raise PatchRunReceiptError("LEDGER_HANDOFF_TIME_INVALID")
    return value


def _validate_structure(handoff: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(handoff, Mapping) or set(handoff) != _HANDOFF_FIELDS:
        raise PatchRunReceiptError("LEDGER_HANDOFF_SCHEMA_INVALID")
    if (
        handoff.get("schema_version") != HANDOFF_SCHEMA
        or handoff.get("algorithm") != HANDOFF_ALGORITHM
        or handoff.get("purpose") != HANDOFF_PURPOSE
    ):
        raise PatchRunReceiptError("LEDGER_HANDOFF_SCHEMA_INVALID")
    _validate_identifier(handoff.get("key_id"), "LEDGER_HANDOFF_KEY_ID_INVALID")
    _validate_identifier(
        handoff.get("source_controller_id"),
        "LEDGER_HANDOFF_SOURCE_INVALID",
    )
    _validate_identifier(handoff.get("standby_id"), "LEDGER_HANDOFF_STANDBY_INVALID")
    _validate_epoch(handoff.get("authority_epoch"))
    issued_at = _validate_timestamp(handoff.get("issued_at_epoch"))
    expires_at = _validate_timestamp(handoff.get("expires_at_epoch"))
    if not issued_at < expires_at <= issued_at + MAX_HANDOFF_LIFETIME_SECONDS:
        raise PatchRunReceiptError("LEDGER_HANDOFF_TIME_INVALID")
    for field in (
        "backup_manifest_sha256",
        "backup_sha256",
        "promotion_manifest_sha256",
        "downstream_fence_sha256",
        "signature",
    ):
        _validate_digest(handoff.get(field), "LEDGER_HANDOFF_DIGEST_INVALID")
    return dict(handoff)


def _strict_json_object(text: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PatchRunReceiptError("LEDGER_HANDOFF_FILE_INVALID")
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=reject_duplicate_keys)
    except PatchRunReceiptError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise PatchRunReceiptError("LEDGER_HANDOFF_FILE_INVALID") from exc
    if not isinstance(value, dict):
        raise PatchRunReceiptError("LEDGER_HANDOFF_FILE_INVALID")
    return value


def load_rank_ledger_handoff(path: Path) -> dict[str, object]:
    """Load a bounded handoff document without following symlinks."""

    try:
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > MAX_HANDOFF_BYTES
        ):
            raise PatchRunReceiptError("LEDGER_HANDOFF_FILE_INVALID")
        payload = path.read_bytes()
        if len(payload) > MAX_HANDOFF_BYTES:
            raise PatchRunReceiptError("LEDGER_HANDOFF_FILE_INVALID")
        document = _strict_json_object(payload.decode("utf-8"))
    except PatchRunReceiptError:
        raise
    except (OSError, UnicodeDecodeError) as exc:
        raise PatchRunReceiptError("LEDGER_HANDOFF_FILE_INVALID") from exc
    return _validate_structure(document)


def create_rank_ledger_handoff(
    backup_path: Path,
    backup_manifest: Mapping[str, object],
    *,
    authority_epoch: int,
    source_controller_id: str,
    standby_id: str,
    downstream_fence_sha256: str,
    issued_at_epoch: int,
    expires_at_epoch: int,
    key_id: str,
    key: bytes,
) -> dict[str, object]:
    """Sign one verified backup for one fenced standby activation epoch."""

    verification = verify_rank_ledger_backup(backup_path, dict(backup_manifest))
    claims: dict[str, object] = {
        "schema_version": HANDOFF_SCHEMA,
        "algorithm": HANDOFF_ALGORITHM,
        "purpose": HANDOFF_PURPOSE,
        "key_id": _validate_identifier(key_id, "LEDGER_HANDOFF_KEY_ID_INVALID"),
        "authority_epoch": _validate_epoch(authority_epoch),
        "source_controller_id": _validate_identifier(
            source_controller_id,
            "LEDGER_HANDOFF_SOURCE_INVALID",
        ),
        "standby_id": _validate_identifier(
            standby_id,
            "LEDGER_HANDOFF_STANDBY_INVALID",
        ),
        "issued_at_epoch": _validate_timestamp(issued_at_epoch),
        "expires_at_epoch": _validate_timestamp(expires_at_epoch),
        "backup_manifest_sha256": verification["manifest_sha256"],
        "backup_sha256": verification["backup_sha256"],
        "promotion_manifest_sha256": verification["promotion_manifest_sha256"],
        "downstream_fence_sha256": _validate_digest(
            downstream_fence_sha256,
            "LEDGER_HANDOFF_FENCE_INVALID",
        ),
    }
    if (
        not claims["issued_at_epoch"]
        < claims["expires_at_epoch"]
        <= (int(claims["issued_at_epoch"]) + MAX_HANDOFF_LIFETIME_SECONDS)
    ):
        raise PatchRunReceiptError("LEDGER_HANDOFF_TIME_INVALID")
    signature = hmac.new(_validate_key(key), _canonical_json(claims), hashlib.sha256).hexdigest()
    return {**claims, "signature": signature}


def verify_rank_ledger_handoff(
    backup_path: Path,
    backup_manifest: Mapping[str, object],
    handoff: Mapping[str, object],
    keyring: Mapping[str, bytes],
    *,
    required_authority_epoch: int,
    expected_source_controller_id: str,
    expected_standby_id: str,
    expected_downstream_fence_sha256: str,
    now_epoch: int,
) -> dict[str, object]:
    """Authenticate a handoff against external epoch, identity and fence state."""

    normalized = _validate_structure(handoff)
    if not isinstance(keyring, Mapping):
        raise PatchRunReceiptError("LEDGER_HANDOFF_KEYRING_INVALID")
    key = keyring.get(str(normalized["key_id"]))
    if key is None:
        raise PatchRunReceiptError("LEDGER_HANDOFF_KEY_UNKNOWN")
    expected_signature = hmac.new(
        _validate_key(key),
        _canonical_json(_claims(normalized)),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(str(normalized["signature"]), expected_signature):
        raise PatchRunReceiptError("LEDGER_HANDOFF_SIGNATURE_INVALID")

    required_epoch = _validate_epoch(required_authority_epoch)
    source_id = _validate_identifier(
        expected_source_controller_id,
        "LEDGER_HANDOFF_SOURCE_INVALID",
    )
    standby_id = _validate_identifier(
        expected_standby_id,
        "LEDGER_HANDOFF_STANDBY_INVALID",
    )
    fence_sha256 = _validate_digest(
        expected_downstream_fence_sha256,
        "LEDGER_HANDOFF_FENCE_INVALID",
    )
    current_time = _validate_timestamp(now_epoch)
    if normalized["authority_epoch"] != required_epoch:
        raise PatchRunReceiptError("LEDGER_HANDOFF_EPOCH_MISMATCH")
    if not hmac.compare_digest(str(normalized["source_controller_id"]), source_id):
        raise PatchRunReceiptError("LEDGER_HANDOFF_SOURCE_MISMATCH")
    if not hmac.compare_digest(str(normalized["standby_id"]), standby_id):
        raise PatchRunReceiptError("LEDGER_HANDOFF_STANDBY_MISMATCH")
    if not hmac.compare_digest(str(normalized["downstream_fence_sha256"]), fence_sha256):
        raise PatchRunReceiptError("LEDGER_HANDOFF_FENCE_MISMATCH")
    if current_time < int(normalized["issued_at_epoch"]) - MAX_CLOCK_SKEW_SECONDS:
        raise PatchRunReceiptError("LEDGER_HANDOFF_NOT_YET_VALID")
    if current_time > int(normalized["expires_at_epoch"]):
        raise PatchRunReceiptError("LEDGER_HANDOFF_EXPIRED")

    verification = verify_rank_ledger_backup(backup_path, dict(backup_manifest))
    if any(
        not hmac.compare_digest(str(normalized[field]), str(verification[expected]))
        for field, expected in (
            ("backup_manifest_sha256", "manifest_sha256"),
            ("backup_sha256", "backup_sha256"),
            ("promotion_manifest_sha256", "promotion_manifest_sha256"),
        )
    ):
        raise PatchRunReceiptError("LEDGER_HANDOFF_BACKUP_MISMATCH")
    return {
        "schema_version": HANDOFF_VERIFICATION_SCHEMA,
        "authorized": True,
        "authority_epoch": required_epoch,
        "source_controller_id": source_id,
        "standby_id": standby_id,
        "backup_sha256": verification["backup_sha256"],
        "promotion_count": verification["promotion_count"],
        "receipt_count": verification["receipt_count"],
        "promotion_manifest_sha256": verification["promotion_manifest_sha256"],
        "backup_manifest_sha256": verification["manifest_sha256"],
        "downstream_fence_sha256": fence_sha256,
        "handoff_sha256": _digest(normalized),
    }


def restore_authorized_rank_ledger(
    backup_path: Path,
    backup_manifest: Mapping[str, object],
    handoff: Mapping[str, object],
    keyring: Mapping[str, bytes],
    ledger_path: Path,
    *,
    required_authority_epoch: int,
    expected_source_controller_id: str,
    expected_standby_id: str,
    expected_downstream_fence_sha256: str,
    now_epoch: int,
) -> dict[str, object]:
    """Authorize first, then restore a new standby ledger without overwriting state."""

    manifest_copy = dict(backup_manifest)
    authorization = verify_rank_ledger_handoff(
        backup_path,
        manifest_copy,
        dict(handoff),
        keyring,
        required_authority_epoch=required_authority_epoch,
        expected_source_controller_id=expected_source_controller_id,
        expected_standby_id=expected_standby_id,
        expected_downstream_fence_sha256=expected_downstream_fence_sha256,
        now_epoch=now_epoch,
    )
    restored = restore_rank_ledger_backup(backup_path, manifest_copy, ledger_path)
    return {
        **restored,
        **authorization,
        "schema_version": AUTHORIZED_RESTORE_SCHEMA,
        "restored": True,
    }
