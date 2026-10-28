from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import sqlite3
import stat
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

from structxai.patch_dose_response import PatchSweepPoint, PatchSweepResult
from structxai.patch_run_receipt import (
    AuthenticatedCrossSeedAudit,
    PatchRunReceipt,
    PatchRunReceiptError,
    audit_authenticated_cross_seed_dose_response,
)

INGESTION_SCHEMA = "struct-xai-rank-receipt-ingestion/v1"
INGESTION_RESULT_SCHEMA = "struct-xai-rank-receipt-ingestion-result/v1"
DEFAULT_KEYRING_ENV = "STRUCTXAI_PATCH_RECEIPT_KEYRING_JSON"
MAX_ENVELOPE_BYTES = 8 * 1024 * 1024
PROMOTION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


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
        raise PatchRunReceiptError("INGESTION_NOT_CANONICAL") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PatchRunReceiptError("INGESTION_DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _load_json(text: str, error_code: str) -> object:
    try:
        return json.loads(text, object_pairs_hook=_object_without_duplicate_keys)
    except PatchRunReceiptError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise PatchRunReceiptError(error_code) from exc


def load_ingestion_envelope(path: Path) -> object:
    """Read a bounded JSON envelope while rejecting symlinks and duplicate keys."""

    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise PatchRunReceiptError("INGESTION_INPUT_INVALID")
        if metadata.st_size > MAX_ENVELOPE_BYTES:
            raise PatchRunReceiptError("INGESTION_INPUT_TOO_LARGE")
        payload = path.read_bytes()
    except PatchRunReceiptError:
        raise
    except OSError as exc:
        raise PatchRunReceiptError("INGESTION_INPUT_UNREADABLE") from exc
    if len(payload) > MAX_ENVELOPE_BYTES:
        raise PatchRunReceiptError("INGESTION_INPUT_TOO_LARGE")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PatchRunReceiptError("INGESTION_INPUT_INVALID") from exc
    return _load_json(text, "INGESTION_INPUT_INVALID")


def load_keyring_from_env(env_name: str = DEFAULT_KEYRING_ENV) -> dict[str, bytes]:
    """Load base64-encoded signing keys from one JSON environment variable."""

    encoded = os.getenv(env_name)
    if not encoded:
        raise PatchRunReceiptError("INGESTION_KEYRING_MISSING")
    payload = _load_json(encoded, "INGESTION_KEYRING_INVALID")
    if not isinstance(payload, dict) or not payload:
        raise PatchRunReceiptError("INGESTION_KEYRING_INVALID")
    keyring: dict[str, bytes] = {}
    for key_id, value in payload.items():
        if not isinstance(key_id, str) or not isinstance(value, str):
            raise PatchRunReceiptError("INGESTION_KEYRING_INVALID")
        try:
            keyring[key_id] = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise PatchRunReceiptError("INGESTION_KEYRING_INVALID") from exc
    return keyring


def _require_exact_fields(payload: object, expected: set[str], code: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise PatchRunReceiptError(code)
    return payload


def _parse_sweep_result(payload: object) -> PatchSweepResult:
    result_fields = {field.name for field in fields(PatchSweepResult)}
    values = _require_exact_fields(payload, result_fields, "INGESTION_SWEEP_SCHEMA_INVALID")
    point_fields = {field.name for field in fields(PatchSweepPoint)}
    raw_points = values.get("points")
    if not isinstance(raw_points, list):
        raise PatchRunReceiptError("INGESTION_SWEEP_SCHEMA_INVALID")
    points = []
    for raw_point in raw_points:
        point = _require_exact_fields(raw_point, point_fields, "INGESTION_SWEEP_SCHEMA_INVALID")
        try:
            points.append(PatchSweepPoint(**point))
        except TypeError as exc:
            raise PatchRunReceiptError("INGESTION_SWEEP_SCHEMA_INVALID") from exc
    try:
        return PatchSweepResult(**{**values, "points": tuple(points)})
    except TypeError as exc:
        raise PatchRunReceiptError("INGESTION_SWEEP_SCHEMA_INVALID") from exc


def _parse_receipt(payload: object) -> PatchRunReceipt:
    receipt_fields = {field.name for field in fields(PatchRunReceipt)}
    values = _require_exact_fields(payload, receipt_fields, "INGESTION_RECEIPT_SCHEMA_INVALID")
    try:
        return PatchRunReceipt(**values)
    except TypeError as exc:
        raise PatchRunReceiptError("INGESTION_RECEIPT_SCHEMA_INVALID") from exc


def _parse_envelope(
    envelope: object,
) -> tuple[str, dict[int, PatchSweepResult], dict[int, dict[int, PatchRunReceipt]], dict[str, object]]:
    root = _require_exact_fields(
        envelope,
        {"schema_version", "promotion_id", "runs"},
        "INGESTION_SCHEMA_INVALID",
    )
    if root["schema_version"] != INGESTION_SCHEMA:
        raise PatchRunReceiptError("INGESTION_SCHEMA_INVALID")
    promotion_id = root["promotion_id"]
    if not isinstance(promotion_id, str) or PROMOTION_ID_RE.fullmatch(promotion_id) is None:
        raise PatchRunReceiptError("PROMOTION_ID_INVALID")
    raw_runs = root["runs"]
    if not isinstance(raw_runs, list) or not raw_runs:
        raise PatchRunReceiptError("INGESTION_RUN_SET_INVALID")

    runs: dict[int, PatchSweepResult] = {}
    rank_receipt_sets: dict[int, dict[int, PatchRunReceipt]] = {}
    canonical_runs: list[dict[str, object]] = []
    for raw_run in raw_runs:
        run = _require_exact_fields(
            raw_run,
            {"seed", "sweep_result", "rank_receipts"},
            "INGESTION_RUN_SCHEMA_INVALID",
        )
        seed = run["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed in runs:
            raise PatchRunReceiptError("INGESTION_SEED_SET_INVALID")
        result = _parse_sweep_result(run["sweep_result"])
        raw_receipts = run["rank_receipts"]
        if not isinstance(raw_receipts, list) or not raw_receipts:
            raise PatchRunReceiptError("RANK_RECEIPT_SET_INVALID")
        rank_receipts: dict[int, PatchRunReceipt] = {}
        for raw_receipt in raw_receipts:
            receipt = _parse_receipt(raw_receipt)
            if receipt.worker_rank in rank_receipts:
                raise PatchRunReceiptError("RANK_RECEIPT_SET_INVALID")
            rank_receipts[receipt.worker_rank] = receipt
        runs[seed] = result
        rank_receipt_sets[seed] = rank_receipts
        canonical_runs.append(
            {
                "seed": seed,
                "sweep_result": result.as_dict(),
                "rank_receipts": [
                    rank_receipts[rank].as_dict() for rank in sorted(rank_receipts)
                ],
            }
        )

    canonical_envelope: dict[str, object] = {
        "schema_version": INGESTION_SCHEMA,
        "promotion_id": promotion_id,
        "runs": sorted(canonical_runs, key=lambda item: int(item["seed"])),
    }
    return promotion_id, runs, rank_receipt_sets, canonical_envelope


def _validate_ledger_path(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if path.exists():
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise PatchRunReceiptError("REPLAY_LEDGER_INVALID")
    except PatchRunReceiptError:
        raise
    except OSError as exc:
        raise PatchRunReceiptError("REPLAY_LEDGER_UNAVAILABLE") from exc


def _open_ledger(path: Path) -> sqlite3.Connection:
    _validate_ledger_path(path)
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(path, isolation_level=None, timeout=30.0)
        os.chmod(path, 0o600)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = FULL")
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version not in (0, 1):
            raise PatchRunReceiptError("REPLAY_LEDGER_VERSION_UNSUPPORTED")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS promotions (
                promotion_id TEXT PRIMARY KEY,
                envelope_sha256 TEXT NOT NULL,
                audit_json TEXT NOT NULL,
                receipt_count INTEGER NOT NULL CHECK (receipt_count > 0)
            ) STRICT
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS consumed_receipts (
                receipt_sha256 TEXT PRIMARY KEY,
                promotion_id TEXT NOT NULL REFERENCES promotions(promotion_id)
            ) STRICT
            """
        )
        if version == 0:
            connection.execute("PRAGMA user_version = 1")
        connection.execute("COMMIT")
        return connection
    except PatchRunReceiptError:
        if connection is not None:
            connection.close()
        raise
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        if connection is not None:
            connection.close()
        raise PatchRunReceiptError("REPLAY_LEDGER_UNAVAILABLE") from exc


def _consume_receipts(
    ledger_path: Path,
    *,
    promotion_id: str,
    envelope_sha256: str,
    receipt_sha256s: tuple[str, ...],
    audit: AuthenticatedCrossSeedAudit,
) -> tuple[dict[str, object], bool]:
    connection = _open_ledger(ledger_path)
    audit_json = _canonical_json(audit.as_dict())
    try:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            "SELECT envelope_sha256, audit_json, receipt_count FROM promotions WHERE promotion_id = ?",
            (promotion_id,),
        ).fetchone()
        if existing is not None:
            if existing[0] != envelope_sha256 or existing[2] != len(receipt_sha256s):
                raise PatchRunReceiptError("PROMOTION_ID_REUSED")
            if existing[1] != audit_json:
                raise PatchRunReceiptError("REPLAY_LEDGER_INVALID")
            stored_audit = _load_json(existing[1], "REPLAY_LEDGER_INVALID")
            connection.execute("COMMIT")
            if not isinstance(stored_audit, dict):
                raise PatchRunReceiptError("REPLAY_LEDGER_INVALID")
            return stored_audit, True

        connection.execute(
            "INSERT INTO promotions VALUES (?, ?, ?, ?)",
            (promotion_id, envelope_sha256, audit_json, len(receipt_sha256s)),
        )
        try:
            connection.executemany(
                "INSERT INTO consumed_receipts VALUES (?, ?)",
                ((digest, promotion_id) for digest in receipt_sha256s),
            )
        except sqlite3.IntegrityError as exc:
            raise PatchRunReceiptError("RANK_RECEIPT_REPLAYED") from exc
        connection.execute("COMMIT")
        return audit.as_dict(), False
    except PatchRunReceiptError:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    except sqlite3.Error as exc:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise PatchRunReceiptError("REPLAY_LEDGER_UNAVAILABLE") from exc
    finally:
        connection.close()


def ingest_rank_receipt_envelope(
    envelope: object,
    keyring: Mapping[str, bytes],
    ledger_path: Path,
    *,
    now_epoch: int | None = None,
) -> dict[str, object]:
    """Verify a promotion envelope and atomically consume every signed rank receipt.

    An exact retry of the same promotion is idempotent so an interrupted artifact
    write can be recovered. Reusing any receipt under a different promotion fails.
    """

    promotion_id, runs, rank_receipt_sets, canonical_envelope = _parse_envelope(envelope)
    coordinator_receipts = {seed: receipts[0] for seed, receipts in rank_receipt_sets.items() if 0 in receipts}
    if set(coordinator_receipts) != set(runs):
        raise PatchRunReceiptError("RANK_ZERO_RECEIPT_MISSING")
    audit = audit_authenticated_cross_seed_dose_response(
        runs,
        coordinator_receipts,
        keyring,
        now_epoch=now_epoch,
        rank_receipt_sets=rank_receipt_sets,
    )
    receipt_sha256s = tuple(
        _digest(rank_receipt_sets[seed][rank].as_dict())
        for seed in sorted(rank_receipt_sets)
        for rank in sorted(rank_receipt_sets[seed])
    )
    envelope_sha256 = _digest(canonical_envelope)
    stored_audit, replayed = _consume_receipts(
        ledger_path,
        promotion_id=promotion_id,
        envelope_sha256=envelope_sha256,
        receipt_sha256s=receipt_sha256s,
        audit=audit,
    )
    return {
        "schema_version": INGESTION_RESULT_SCHEMA,
        "promotion_id": promotion_id,
        "envelope_sha256": envelope_sha256,
        "receipt_count": len(receipt_sha256s),
        "replayed": replayed,
        "audit": stored_audit,
    }
