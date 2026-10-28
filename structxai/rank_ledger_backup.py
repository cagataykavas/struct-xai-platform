from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from structxai.patch_run_receipt import PatchRunReceiptError

BACKUP_MANIFEST_SCHEMA = "struct-xai-rank-ledger-backup/v1"
BACKUP_VERIFICATION_SCHEMA = "struct-xai-rank-ledger-backup-verification/v1"
LEDGER_USER_VERSION = 1
MAX_LEDGER_BYTES = 1024 * 1024 * 1024
MAX_MANIFEST_BYTES = 64 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_FIELDS = {
    "schema_version",
    "ledger_user_version",
    "backup_sha256",
    "promotion_count",
    "receipt_count",
    "promotion_manifest_sha256",
    "manifest_sha256",
}
_PROMOTION_COLUMNS = (
    ("promotion_id", "TEXT", 1, 1),
    ("envelope_sha256", "TEXT", 1, 0),
    ("audit_json", "TEXT", 1, 0),
    ("receipt_count", "INTEGER", 1, 0),
)
_RECEIPT_COLUMNS = (
    ("receipt_sha256", "TEXT", 1, 1),
    ("promotion_id", "TEXT", 1, 0),
)


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise PatchRunReceiptError("LEDGER_BACKUP_NOT_CANONICAL") from exc


def _digest_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _require_regular_file(path: Path, *, maximum_bytes: int, code: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise PatchRunReceiptError(code) from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size > maximum_bytes
    ):
        raise PatchRunReceiptError(code)
    return metadata


def _file_sha256(path: Path) -> str:
    _require_regular_file(
        path,
        maximum_bytes=MAX_LEDGER_BYTES,
        code="LEDGER_BACKUP_INVALID",
    )
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise PatchRunReceiptError("LEDGER_BACKUP_UNAVAILABLE") from exc
    return digest.hexdigest()


def _validate_parent_and_target(path: Path, *, exists_code: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if _path_exists(path):
            raise PatchRunReceiptError(exists_code)
    except PatchRunReceiptError:
        raise
    except OSError as exc:
        raise PatchRunReceiptError("LEDGER_BACKUP_UNAVAILABLE") from exc


def _publish_new_file(temporary: Path, target: Path, *, exists_code: str) -> None:
    try:
        os.link(temporary, target)
        temporary.unlink()
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as exc:
        raise PatchRunReceiptError(exists_code) from exc
    except OSError as exc:
        raise PatchRunReceiptError("LEDGER_BACKUP_UNAVAILABLE") from exc


def _strict_json_object(text: str, code: str) -> dict[str, Any]:
    def object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PatchRunReceiptError(code)
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=object_without_duplicate_keys)
    except PatchRunReceiptError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise PatchRunReceiptError(code) from exc
    if not isinstance(value, dict):
        raise PatchRunReceiptError(code)
    return value


def load_rank_ledger_backup_manifest(path: Path) -> dict[str, object]:
    """Load a bounded, strict backup manifest without following symlinks."""

    _require_regular_file(
        path,
        maximum_bytes=MAX_MANIFEST_BYTES,
        code="LEDGER_BACKUP_MANIFEST_INVALID",
    )
    try:
        payload = path.read_bytes()
        text = payload.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise PatchRunReceiptError("LEDGER_BACKUP_MANIFEST_INVALID") from exc
    return _strict_json_object(text, "LEDGER_BACKUP_MANIFEST_INVALID")


def _open_read_only(path: Path, code: str) -> sqlite3.Connection:
    _require_regular_file(path, maximum_bytes=MAX_LEDGER_BYTES, code=code)
    try:
        connection = sqlite3.connect(
            f"{path.resolve().as_uri()}?mode=ro",
            uri=True,
            isolation_level=None,
            timeout=30.0,
        )
        connection.execute("PRAGMA query_only = ON")
        return connection
    except (OSError, sqlite3.Error, ValueError) as exc:
        raise PatchRunReceiptError(code) from exc


def _table_columns(connection: sqlite3.Connection, table: str) -> tuple[tuple[object, ...], ...]:
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    return tuple((row[1], row[2], row[3], row[5]) for row in rows)


def _inspect_connection(connection: sqlite3.Connection) -> dict[str, object]:
    try:
        if int(connection.execute("PRAGMA user_version").fetchone()[0]) != LEDGER_USER_VERSION:
            raise PatchRunReceiptError("LEDGER_BACKUP_VERSION_UNSUPPORTED")
        integrity = connection.execute("PRAGMA integrity_check").fetchall()
        if integrity != [("ok",)] or connection.execute("PRAGMA foreign_key_check").fetchone():
            raise PatchRunReceiptError("LEDGER_BACKUP_INTEGRITY_FAILED")
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        if tables != {"promotions", "consumed_receipts"}:
            raise PatchRunReceiptError("LEDGER_BACKUP_SCHEMA_INVALID")
        if (
            _table_columns(connection, "promotions") != _PROMOTION_COLUMNS
            or _table_columns(connection, "consumed_receipts") != _RECEIPT_COLUMNS
        ):
            raise PatchRunReceiptError("LEDGER_BACKUP_SCHEMA_INVALID")

        promotion_rows: list[dict[str, object]] = []
        total_receipts = 0
        promotions = connection.execute(
            "SELECT promotion_id, envelope_sha256, audit_json, receipt_count "
            "FROM promotions ORDER BY promotion_id"
        ).fetchall()
        for promotion_id, envelope_sha256, audit_json, receipt_count in promotions:
            if (
                not isinstance(promotion_id, str)
                or not isinstance(envelope_sha256, str)
                or _SHA256_RE.fullmatch(envelope_sha256) is None
                or not isinstance(audit_json, str)
                or type(receipt_count) is not int
                or receipt_count <= 0
            ):
                raise PatchRunReceiptError("LEDGER_BACKUP_ROW_INVALID")
            audit = _strict_json_object(audit_json, "LEDGER_BACKUP_ROW_INVALID")
            audit_evidence_sha256 = audit.get("evidence_sha256")
            audit_body = {key: value for key, value in audit.items() if key != "evidence_sha256"}
            if (
                _canonical_json(audit) != audit_json
                or audit.get("schema_version") != "struct-xai-authenticated-cross-seed/v2"
                or not isinstance(audit_evidence_sha256, str)
                or _SHA256_RE.fullmatch(audit_evidence_sha256) is None
                or _digest_json(audit_body) != audit_evidence_sha256
            ):
                raise PatchRunReceiptError("LEDGER_BACKUP_ROW_INVALID")
            receipt_rows = connection.execute(
                "SELECT receipt_sha256 FROM consumed_receipts WHERE promotion_id = ? ORDER BY receipt_sha256",
                (promotion_id,),
            ).fetchall()
            receipts = [row[0] for row in receipt_rows]
            if len(receipts) != receipt_count or any(
                not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None for digest in receipts
            ):
                raise PatchRunReceiptError("LEDGER_BACKUP_ROW_INVALID")
            total_receipts += len(receipts)
            promotion_rows.append(
                {
                    "promotion_id_sha256": hashlib.sha256(promotion_id.encode("utf-8")).hexdigest(),
                    "envelope_sha256": envelope_sha256,
                    "audit_sha256": hashlib.sha256(audit_json.encode("ascii")).hexdigest(),
                    "audit_evidence_sha256": audit_evidence_sha256,
                    "receipt_count": receipt_count,
                    "receipt_manifest_sha256": _digest_json(receipts),
                }
            )
        orphan_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM consumed_receipts r "
                "LEFT JOIN promotions p ON p.promotion_id = r.promotion_id "
                "WHERE p.promotion_id IS NULL"
            ).fetchone()[0]
        )
        if orphan_count:
            raise PatchRunReceiptError("LEDGER_BACKUP_INTEGRITY_FAILED")
        return {
            "promotion_count": len(promotion_rows),
            "receipt_count": total_receipts,
            "promotion_manifest_sha256": _digest_json(promotion_rows),
        }
    except PatchRunReceiptError:
        raise
    except (UnicodeEncodeError, sqlite3.Error, TypeError, ValueError) as exc:
        raise PatchRunReceiptError("LEDGER_BACKUP_INVALID") from exc


def _inspect_ledger(path: Path, code: str = "LEDGER_BACKUP_INVALID") -> dict[str, object]:
    connection = _open_read_only(path, code)
    try:
        return _inspect_connection(connection)
    finally:
        connection.close()


def _validate_manifest(manifest: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(manifest, Mapping) or set(manifest) != _MANIFEST_FIELDS:
        raise PatchRunReceiptError("LEDGER_BACKUP_MANIFEST_INVALID")
    document = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        manifest.get("schema_version") != BACKUP_MANIFEST_SCHEMA
        or manifest.get("ledger_user_version") != LEDGER_USER_VERSION
        or type(manifest.get("promotion_count")) is not int
        or int(manifest["promotion_count"]) < 0
        or type(manifest.get("receipt_count")) is not int
        or int(manifest["receipt_count"]) < 0
        or any(
            not isinstance(manifest.get(name), str) or _SHA256_RE.fullmatch(str(manifest[name])) is None
            for name in ("backup_sha256", "promotion_manifest_sha256", "manifest_sha256")
        )
        or manifest["manifest_sha256"] != _digest_json(document)
    ):
        raise PatchRunReceiptError("LEDGER_BACKUP_MANIFEST_INVALID")
    return dict(manifest)


def _write_manifest_new(manifest: Mapping[str, object], path: Path) -> None:
    _validate_parent_and_target(path, exists_code="LEDGER_BACKUP_MANIFEST_EXISTS")
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(json.dumps(manifest, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _publish_new_file(temporary, path, exists_code="LEDGER_BACKUP_MANIFEST_EXISTS")
        temporary = None
    except PatchRunReceiptError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise PatchRunReceiptError("LEDGER_BACKUP_UNAVAILABLE") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def create_rank_ledger_backup(
    ledger_path: Path,
    backup_path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    """Create an online SQLite snapshot plus a privacy-reduced integrity manifest."""

    _validate_parent_and_target(backup_path, exists_code="LEDGER_BACKUP_EXISTS")
    if _path_exists(manifest_path):
        raise PatchRunReceiptError("LEDGER_BACKUP_MANIFEST_EXISTS")
    source = _open_read_only(ledger_path, "REPLAY_LEDGER_INVALID")
    temporary: Path | None = None
    published = False
    try:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{backup_path.name}.", dir=backup_path.parent)
        os.close(descriptor)
        temporary = Path(temporary_name)
        os.chmod(temporary, 0o600)
        destination = sqlite3.connect(temporary)
        try:
            source.backup(destination)
        finally:
            destination.close()
        details = _inspect_ledger(temporary)
        _publish_new_file(temporary, backup_path, exists_code="LEDGER_BACKUP_EXISTS")
        temporary = None
        published = True
        document = {
            "schema_version": BACKUP_MANIFEST_SCHEMA,
            "ledger_user_version": LEDGER_USER_VERSION,
            "backup_sha256": _file_sha256(backup_path),
            **details,
        }
        manifest = {**document, "manifest_sha256": _digest_json(document)}
        _write_manifest_new(manifest, manifest_path)
        return manifest
    except PatchRunReceiptError:
        if published:
            backup_path.unlink(missing_ok=True)
        raise
    except (OSError, sqlite3.Error) as exc:
        if published:
            backup_path.unlink(missing_ok=True)
        raise PatchRunReceiptError("LEDGER_BACKUP_UNAVAILABLE") from exc
    finally:
        source.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_rank_ledger_backup(
    backup_path: Path,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Verify physical bytes, SQLite integrity, schema and logical promotion state."""

    normalized = _validate_manifest(manifest)
    backup_sha256 = _file_sha256(backup_path)
    if backup_sha256 != normalized["backup_sha256"]:
        raise PatchRunReceiptError("LEDGER_BACKUP_DIGEST_MISMATCH")
    details = _inspect_ledger(backup_path)
    if any(details[name] != normalized[name] for name in details):
        raise PatchRunReceiptError("LEDGER_BACKUP_STATE_MISMATCH")
    return {
        "schema_version": BACKUP_VERIFICATION_SCHEMA,
        "verified": True,
        "backup_sha256": backup_sha256,
        "promotion_count": details["promotion_count"],
        "receipt_count": details["receipt_count"],
        "promotion_manifest_sha256": details["promotion_manifest_sha256"],
        "manifest_sha256": normalized["manifest_sha256"],
    }


def restore_rank_ledger_backup(
    backup_path: Path,
    manifest: Mapping[str, object],
    ledger_path: Path,
) -> dict[str, object]:
    """Verify and publish a new standby ledger without overwriting existing state."""

    verification = verify_rank_ledger_backup(backup_path, manifest)
    _validate_parent_and_target(ledger_path, exists_code="LEDGER_RESTORE_TARGET_EXISTS")
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{ledger_path.name}.", dir=ledger_path.parent)
        temporary = Path(temporary_name)
        os.fchmod(descriptor, 0o600)
        with backup_path.open("rb") as source, os.fdopen(descriptor, "wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
        if _file_sha256(temporary) != verification["backup_sha256"]:
            raise PatchRunReceiptError("LEDGER_BACKUP_DIGEST_MISMATCH")
        _publish_new_file(temporary, ledger_path, exists_code="LEDGER_RESTORE_TARGET_EXISTS")
        temporary = None
        restored = verify_rank_ledger_backup(ledger_path, manifest)
        return {**restored, "schema_version": BACKUP_VERIFICATION_SCHEMA, "restored": True}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
