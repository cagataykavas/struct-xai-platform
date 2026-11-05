from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from structxai.stability_manifest import ExternalEvidenceVerifier
from structxai.stability_reference_custody import (
    MAX_CUSTODY_CLOCK_SKEW_SECONDS,
    StabilityReferenceCustodyError,
    verify_custody_key_status_snapshot,
)

STATUS_TRANSPARENCY_ENTRY_SCHEMA = "struct-xai-custody-status-transparency-entry/v1"
STATUS_TRANSPARENCY_CHECKPOINT_SCHEMA = "struct-xai-custody-status-transparency-checkpoint/v1"
MAX_STATUS_TRANSPARENCY_LOG_BYTES = 16 * 1024 * 1024
MAX_STATUS_TRANSPARENCY_ENTRIES = 100_000

_ZERO_DIGEST = "0" * 64
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_ENTRY_FIELDS = {
    "schema_version",
    "log_id",
    "sequence",
    "snapshot_sha256",
    "verification_sha256",
    "observed_at_epoch",
    "previous_entry_sha256",
    "entry_sha256",
}
_CHECKPOINT_FIELDS = {
    "schema_version",
    "log_id",
    "tree_size",
    "head_entry_sha256",
    "checkpoint_sha256",
}


class CustodyStatusTransparencyError(ValueError):
    """Stable failure from the local custody-status transparency boundary."""

    code = "CUSTODY_STATUS_TRANSPARENCY_REJECTED"

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
        raise CustodyStatusTransparencyError("transparency evidence is not canonical JSON") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _same(left: object, right: object) -> bool:
    return isinstance(left, str) and isinstance(right, str) and hmac.compare_digest(left, right)


def _require_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise CustodyStatusTransparencyError(f"{name} is invalid")
    return value


def _require_log_id(value: object) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise CustodyStatusTransparencyError("log_id is invalid")
    return value


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CustodyStatusTransparencyError("transparency entry contains a duplicate field")
        result[key] = value
    return result


def _entry_claims(entry: Mapping[str, object]) -> dict[str, object]:
    return {key: entry[key] for key in _ENTRY_FIELDS if key != "entry_sha256"}


def _validate_entry(
    value: object,
    *,
    log_id: str,
    expected_sequence: int,
    expected_previous: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _ENTRY_FIELDS:
        raise CustodyStatusTransparencyError("transparency entry must use the exact schema")
    entry = dict(value)
    if entry.get("schema_version") != STATUS_TRANSPARENCY_ENTRY_SCHEMA:
        raise CustodyStatusTransparencyError("transparency entry schema is unsupported")
    if entry.get("log_id") != log_id:
        raise CustodyStatusTransparencyError("transparency entry log identity mismatch")
    if type(entry.get("sequence")) is not int or entry["sequence"] != expected_sequence:
        raise CustodyStatusTransparencyError("transparency entry sequence is invalid")
    if type(entry.get("observed_at_epoch")) is not int or int(entry["observed_at_epoch"]) < 0:
        raise CustodyStatusTransparencyError("transparency entry epoch is invalid")
    _require_digest(entry.get("snapshot_sha256"), "snapshot_sha256")
    _require_digest(entry.get("verification_sha256"), "verification_sha256")
    previous = _require_digest(entry.get("previous_entry_sha256"), "previous_entry_sha256")
    if not hmac.compare_digest(previous, expected_previous):
        raise CustodyStatusTransparencyError("transparency entry chain is inconsistent")
    claimed = _require_digest(entry.get("entry_sha256"), "entry_sha256")
    if not hmac.compare_digest(claimed, _digest(_entry_claims(entry))):
        raise CustodyStatusTransparencyError("transparency entry digest does not match")
    return entry


def _load_entries(payload: bytes, *, log_id: str) -> list[dict[str, object]]:
    if not payload:
        return []
    if len(payload) > MAX_STATUS_TRANSPARENCY_LOG_BYTES or not payload.endswith(b"\n"):
        raise CustodyStatusTransparencyError("transparency log size or termination is invalid")
    lines = payload[:-1].split(b"\n")
    if not 1 <= len(lines) <= MAX_STATUS_TRANSPARENCY_ENTRIES or any(not line for line in lines):
        raise CustodyStatusTransparencyError("transparency log entry count is invalid")
    entries: list[dict[str, object]] = []
    previous = _ZERO_DIGEST
    snapshots: set[str] = set()
    for sequence, line in enumerate(lines, start=1):
        try:
            value = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_fields,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    CustodyStatusTransparencyError(f"invalid JSON constant: {constant}")
                ),
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CustodyStatusTransparencyError("transparency entry is not valid JSON") from exc
        entry = _validate_entry(
            value,
            log_id=log_id,
            expected_sequence=sequence,
            expected_previous=previous,
        )
        if _canonical_json(entry) != line:
            raise CustodyStatusTransparencyError("transparency entry JSON is not canonical")
        snapshot_sha256 = str(entry["snapshot_sha256"])
        if snapshot_sha256 in snapshots:
            raise CustodyStatusTransparencyError("transparency log repeats a snapshot")
        snapshots.add(snapshot_sha256)
        entries.append(entry)
        previous = str(entry["entry_sha256"])
    return entries


def _checkpoint(log_id: str, entries: list[dict[str, object]]) -> dict[str, object]:
    claims: dict[str, object] = {
        "schema_version": STATUS_TRANSPARENCY_CHECKPOINT_SCHEMA,
        "log_id": log_id,
        "tree_size": len(entries),
        "head_entry_sha256": str(entries[-1]["entry_sha256"]) if entries else _ZERO_DIGEST,
    }
    return {**claims, "checkpoint_sha256": _digest(claims)}


def _validate_checkpoint(value: Mapping[str, object], *, log_id: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _CHECKPOINT_FIELDS:
        raise CustodyStatusTransparencyError("prior checkpoint must use the exact schema")
    checkpoint = dict(value)
    if (
        checkpoint.get("schema_version") != STATUS_TRANSPARENCY_CHECKPOINT_SCHEMA
        or checkpoint.get("log_id") != log_id
        or type(checkpoint.get("tree_size")) is not int
        or not 0 <= int(checkpoint["tree_size"]) <= MAX_STATUS_TRANSPARENCY_ENTRIES
    ):
        raise CustodyStatusTransparencyError("prior checkpoint metadata is invalid")
    _require_digest(checkpoint.get("head_entry_sha256"), "head_entry_sha256")
    claimed = _require_digest(checkpoint.get("checkpoint_sha256"), "checkpoint_sha256")
    claims = {key: checkpoint[key] for key in _CHECKPOINT_FIELDS if key != "checkpoint_sha256"}
    if not hmac.compare_digest(claimed, _digest(claims)):
        raise CustodyStatusTransparencyError("prior checkpoint digest does not match")
    return checkpoint


def _read_locked(descriptor: int) -> bytes:
    size = os.fstat(descriptor).st_size
    if size > MAX_STATUS_TRANSPARENCY_LOG_BYTES:
        raise CustodyStatusTransparencyError("transparency log exceeds the byte limit")
    os.lseek(descriptor, 0, os.SEEK_SET)
    payload = bytearray()
    while len(payload) < size:
        chunk = os.read(descriptor, min(1_048_576, size - len(payload)))
        if not chunk:
            raise CustodyStatusTransparencyError("transparency log changed while reading")
        payload.extend(chunk)
    return bytes(payload)


def _open_log(path: Path, *, create: bool) -> int:
    flags = os.O_RDWR if create else os.O_RDONLY
    if create:
        flags |= os.O_CREAT
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise CustodyStatusTransparencyError("transparency log cannot be opened safely") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise CustodyStatusTransparencyError("transparency log permissions are unsafe")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _append_and_sync(descriptor: int, payload: bytes) -> None:
    start_size = os.fstat(descriptor).st_size
    os.lseek(descriptor, 0, os.SEEK_END)
    written = 0
    try:
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("append made no progress")
            written += count
        os.fsync(descriptor)
    except OSError as exc:
        try:
            os.ftruncate(descriptor, start_size)
            os.fsync(descriptor)
        except OSError as rollback_exc:
            raise CustodyStatusTransparencyError(
                "transparency log append failed and could not be rolled back"
            ) from rollback_exc
        raise CustodyStatusTransparencyError("transparency log append failed") from exc


def verify_custody_status_transparency_log(
    path: str | os.PathLike[str],
    *,
    log_id: str,
    prior_checkpoint: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Verify the complete chain and optionally prove extension from a retained checkpoint."""

    normalized_log_id = _require_log_id(log_id)
    descriptor = _open_log(Path(path), create=False)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        entries = _load_entries(_read_locked(descriptor), log_id=normalized_log_id)
    finally:
        os.close(descriptor)
    if prior_checkpoint is not None:
        prior = _validate_checkpoint(prior_checkpoint, log_id=normalized_log_id)
        prior_size = int(prior["tree_size"])
        if prior_size > len(entries):
            raise CustodyStatusTransparencyError("transparency log was truncated")
        observed_head = _ZERO_DIGEST if prior_size == 0 else str(entries[prior_size - 1]["entry_sha256"])
        if not hmac.compare_digest(str(prior["head_entry_sha256"]), observed_head):
            raise CustodyStatusTransparencyError("transparency log is inconsistent with prior checkpoint")
    return _checkpoint(normalized_log_id, entries)


def append_verified_custody_key_status_snapshot(
    path: str | os.PathLike[str],
    snapshot: Mapping[str, object],
    expected_keys: Mapping[str, str],
    authority_verifiers: Mapping[str, ExternalEvidenceVerifier],
    *,
    log_id: str,
    required_authority_signatures: int = 2,
    verification_epoch: int,
    clock_skew_seconds: int = MAX_CUSTODY_CLOCK_SKEW_SECONDS,
) -> tuple[dict[str, object], bool]:
    """Verify a threshold-signed status snapshot and append its commitments exactly once."""

    normalized_log_id = _require_log_id(log_id)
    try:
        verification = verify_custody_key_status_snapshot(
            snapshot,
            expected_keys,
            authority_verifiers,
            required_authority_signatures=required_authority_signatures,
            verification_epoch=verification_epoch,
            clock_skew_seconds=clock_skew_seconds,
        )
    except (StabilityReferenceCustodyError, TypeError, ValueError) as exc:
        raise CustodyStatusTransparencyError("custody key status verification failed") from exc

    descriptor = _open_log(Path(path), create=True)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        entries = _load_entries(_read_locked(descriptor), log_id=normalized_log_id)
        snapshot_sha256 = _require_digest(verification.get("snapshot_sha256"), "snapshot_sha256")
        if any(_same(entry["snapshot_sha256"], snapshot_sha256) for entry in entries):
            return _checkpoint(normalized_log_id, entries), False
        if len(entries) >= MAX_STATUS_TRANSPARENCY_ENTRIES:
            raise CustodyStatusTransparencyError("transparency log entry limit is exhausted")
        claims: dict[str, object] = {
            "schema_version": STATUS_TRANSPARENCY_ENTRY_SCHEMA,
            "log_id": normalized_log_id,
            "sequence": len(entries) + 1,
            "snapshot_sha256": snapshot_sha256,
            "verification_sha256": _require_digest(
                verification.get("verification_sha256"), "verification_sha256"
            ),
            "observed_at_epoch": verification_epoch,
            "previous_entry_sha256": (str(entries[-1]["entry_sha256"]) if entries else _ZERO_DIGEST),
        }
        entry = {**claims, "entry_sha256": _digest(claims)}
        payload = _canonical_json(entry) + b"\n"
        if os.fstat(descriptor).st_size + len(payload) > MAX_STATUS_TRANSPARENCY_LOG_BYTES:
            raise CustodyStatusTransparencyError("transparency log byte limit is exhausted")
        _append_and_sync(descriptor, payload)
        entries.append(entry)
        return _checkpoint(normalized_log_id, entries), True
    finally:
        os.close(descriptor)
