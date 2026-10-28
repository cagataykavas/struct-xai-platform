import base64
import json
import sqlite3
import stat
import time
from pathlib import Path

import pytest

from structxai.cli import main
from structxai.patch_dose_response import PatchSweepPoint, PatchSweepResult
from structxai.patch_run_receipt import PatchRunReceiptError, PatchRunSigner
from structxai.rank_ledger_backup import (
    BACKUP_MANIFEST_SCHEMA,
    BACKUP_VERIFICATION_SCHEMA,
    create_rank_ledger_backup,
    load_rank_ledger_backup_manifest,
    restore_rank_ledger_backup,
    verify_rank_ledger_backup,
)
from structxai.rank_receipt_ingestion import (
    INGESTION_SCHEMA,
    ingest_rank_receipt_envelope,
    load_ingestion_envelope,
)

KEY = b"k" * 32
NOW = int(time.time())
MODEL_DIGEST = "a" * 64
TOKENIZER_DIGEST = "b" * 64
SEEDS = (11, 22, 33)


def sweep(endpoint: float) -> PatchSweepResult:
    target_margin = -1.0
    recoveries = (0.0, endpoint / 2.0, endpoint)
    points = tuple(
        PatchSweepPoint(
            layer=2,
            alpha=alpha,
            positive_score=target_margin + recovery * 2.0,
            negative_score=0.0,
            margin=target_margin + recovery * 2.0,
            margin_delta=recovery * 2.0,
            normalized_recovery=recovery,
        )
        for alpha, recovery in zip((0.0, 0.5, 1.0), recoveries, strict=True)
    )
    return PatchSweepResult(
        schema_version=1,
        model_revision="sha256:test-model",
        source_input_sha256="c" * 64,
        target_input_sha256="d" * 64,
        sweep_spec_sha256="e" * 64,
        policy_sha256="f" * 64,
        source_margin=1.0,
        target_margin=target_margin,
        repeated_target_margin=target_margin,
        model_calls=3 + len(points),
        points=points,
        evidence_sha256="",
    ).with_digest()


def build_envelope(promotion_id: str, *, nonce: int = 0) -> dict[str, object]:
    runs = []
    for seed, endpoint in zip(SEEDS, (0.70, 0.72, 0.68), strict=True):
        result = sweep(endpoint)
        receipts = []
        for rank in range(2):
            run_id = f"{nonce:08x}{seed:012x}{rank:012x}"
            signer = PatchRunSigner(
                "primary",
                KEY,
                clock=lambda: NOW,
                run_id_factory=lambda active_run_id=run_id: active_run_id,
            )
            _, receipt, _manifest = signer.run_with_acknowledged_input_plan(
                lambda initializer, _sampler, active=result: (initializer(0), active)[1],
                seed=seed,
                model_artifact_sha256=MODEL_DIGEST,
                tokenizer_artifact_sha256=TOKENIZER_DIGEST,
                world_size=2,
                rank=rank,
                workers_per_rank=1,
                dataset_size=8,
                epoch=3,
            )
            receipts.append(receipt.as_dict())
        runs.append(
            {
                "seed": seed,
                "sweep_result": result.as_dict(),
                "rank_receipts": receipts,
            }
        )
    return {
        "schema_version": INGESTION_SCHEMA,
        "promotion_id": promotion_id,
        "runs": runs,
    }


@pytest.fixture(scope="module")
def envelope() -> dict[str, object]:
    return build_envelope("release-2026-10-28")


def test_ingestion_emits_v2_audit_and_persists_receipt_claims(
    tmp_path: Path,
    envelope: dict[str, object],
) -> None:
    ledger = tmp_path / "state" / "receipt-ledger.sqlite3"

    result = ingest_rank_receipt_envelope(
        envelope,
        {"primary": KEY},
        ledger,
        now_epoch=NOW,
    )

    assert result["replayed"] is False
    assert result["receipt_count"] == 6
    assert result["audit"]["schema_version"] == "struct-xai-authenticated-cross-seed/v2"
    assert result["audit"]["rank_acknowledgment_bound"] is True
    assert len(result["audit"]["seed_rank_manifests"]) == 3
    assert stat.S_IMODE(ledger.stat().st_mode) == 0o600
    with sqlite3.connect(ledger) as connection:
        assert connection.execute("SELECT COUNT(*) FROM promotions").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM consumed_receipts").fetchone()[0] == 6


def test_exact_promotion_retry_is_idempotent_for_output_recovery(
    tmp_path: Path,
    envelope: dict[str, object],
) -> None:
    ledger = tmp_path / "receipt-ledger.sqlite3"
    first = ingest_rank_receipt_envelope(envelope, {"primary": KEY}, ledger, now_epoch=NOW)
    retry = ingest_rank_receipt_envelope(envelope, {"primary": KEY}, ledger, now_epoch=NOW)

    assert retry["replayed"] is True
    assert retry["audit"] == first["audit"]
    assert retry["envelope_sha256"] == first["envelope_sha256"]
    with sqlite3.connect(ledger) as connection:
        assert connection.execute("SELECT COUNT(*) FROM consumed_receipts").fetchone()[0] == 6


def test_receipts_cannot_be_reused_by_another_promotion(
    tmp_path: Path,
    envelope: dict[str, object],
) -> None:
    ledger = tmp_path / "receipt-ledger.sqlite3"
    ingest_rank_receipt_envelope(envelope, {"primary": KEY}, ledger, now_epoch=NOW)
    reused = {**envelope, "promotion_id": "release-2026-10-29"}

    with pytest.raises(PatchRunReceiptError, match="RANK_RECEIPT_REPLAYED"):
        ingest_rank_receipt_envelope(reused, {"primary": KEY}, ledger, now_epoch=NOW)

    with sqlite3.connect(ledger) as connection:
        assert connection.execute("SELECT COUNT(*) FROM promotions").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM consumed_receipts").fetchone()[0] == 6


def test_promotion_id_cannot_be_rebound_to_new_valid_receipts(
    tmp_path: Path,
    envelope: dict[str, object],
) -> None:
    ledger = tmp_path / "receipt-ledger.sqlite3"
    ingest_rank_receipt_envelope(envelope, {"primary": KEY}, ledger, now_epoch=NOW)
    replacement = build_envelope("release-2026-10-28", nonce=1)

    with pytest.raises(PatchRunReceiptError, match="PROMOTION_ID_REUSED"):
        ingest_rank_receipt_envelope(replacement, {"primary": KEY}, ledger, now_epoch=NOW)


def test_invalid_rank_receipt_never_mutates_replay_state(
    tmp_path: Path,
    envelope: dict[str, object],
) -> None:
    ledger = tmp_path / "receipt-ledger.sqlite3"
    tampered = json.loads(json.dumps(envelope))
    tampered["runs"][0]["rank_receipts"][1]["signature"] = "0" * 64

    with pytest.raises(PatchRunReceiptError, match="RECEIPT_SIGNATURE_INVALID"):
        ingest_rank_receipt_envelope(tampered, {"primary": KEY}, ledger, now_epoch=NOW)

    assert not ledger.exists()


def test_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    envelope_path = tmp_path / "envelope.json"
    envelope_path.write_text(
        '{"schema_version":"struct-xai-rank-receipt-ingestion/v1",'
        '"promotion_id":"one","promotion_id":"two","runs":[]}',
        encoding="utf-8",
    )

    with pytest.raises(PatchRunReceiptError, match="INGESTION_DUPLICATE_JSON_KEY"):
        load_ingestion_envelope(envelope_path)


def test_cli_uses_environment_keyring_and_writes_recoverable_result(
    tmp_path: Path,
    envelope: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope_path = tmp_path / "envelope.json"
    ledger_path = tmp_path / "ledger.sqlite3"
    output_path = tmp_path / "audit.json"
    envelope_path.write_text(json.dumps(envelope), encoding="utf-8")
    monkeypatch.setenv(
        "TEST_STRUCTXAI_KEYRING",
        json.dumps({"primary": base64.b64encode(KEY).decode("ascii")}),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "struct-xai",
            "ingest-rank-receipts",
            str(envelope_path),
            "--ledger",
            str(ledger_path),
            "--output",
            str(output_path),
            "--keyring-env",
            "TEST_STRUCTXAI_KEYRING",
        ],
    )

    assert main() == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["audit"]["schema_version"] == "struct-xai-authenticated-cross-seed/v2"
    assert result["replayed"] is False
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o600


def _ingested_ledger(tmp_path: Path, envelope: dict[str, object]) -> Path:
    ledger = tmp_path / "source" / "receipt-ledger.sqlite3"
    ingest_rank_receipt_envelope(envelope, {"primary": KEY}, ledger, now_epoch=NOW)
    return ledger


def test_online_backup_verifies_logical_state_without_exposing_identifiers(
    tmp_path: Path,
    envelope: dict[str, object],
) -> None:
    ledger = _ingested_ledger(tmp_path, envelope)
    backup = tmp_path / "backup" / "rank-ledger.sqlite3"
    manifest_path = tmp_path / "backup" / "rank-ledger.manifest.json"

    manifest = create_rank_ledger_backup(ledger, backup, manifest_path)
    verification = verify_rank_ledger_backup(backup, manifest)

    assert manifest["schema_version"] == BACKUP_MANIFEST_SCHEMA
    assert manifest["promotion_count"] == 1
    assert manifest["receipt_count"] == 6
    assert verification["schema_version"] == BACKUP_VERIFICATION_SCHEMA
    assert verification["verified"] is True
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600
    serialized = json.dumps(manifest, sort_keys=True)
    assert envelope["promotion_id"] not in serialized
    assert "audit" not in serialized
    assert "signature" not in serialized
    assert load_rank_ledger_backup_manifest(manifest_path) == manifest


def test_verified_restore_preserves_replay_claims(
    tmp_path: Path,
    envelope: dict[str, object],
) -> None:
    ledger = _ingested_ledger(tmp_path, envelope)
    backup = tmp_path / "backup.sqlite3"
    manifest_path = tmp_path / "backup.json"
    manifest = create_rank_ledger_backup(ledger, backup, manifest_path)
    restored = tmp_path / "standby" / "receipt-ledger.sqlite3"

    result = restore_rank_ledger_backup(backup, manifest, restored)
    retry = ingest_rank_receipt_envelope(envelope, {"primary": KEY}, restored, now_epoch=NOW)

    assert result["restored"] is True
    assert result["verified"] is True
    assert retry["replayed"] is True
    reused = {**envelope, "promotion_id": "release-2026-10-29"}
    with pytest.raises(PatchRunReceiptError, match="RANK_RECEIPT_REPLAYED"):
        ingest_rank_receipt_envelope(reused, {"primary": KEY}, restored, now_epoch=NOW)


def test_backup_and_manifest_tampering_fail_closed(
    tmp_path: Path,
    envelope: dict[str, object],
) -> None:
    ledger = _ingested_ledger(tmp_path, envelope)
    backup = tmp_path / "backup.sqlite3"
    manifest_path = tmp_path / "backup.json"
    manifest = create_rank_ledger_backup(ledger, backup, manifest_path)

    tampered_manifest = {**manifest, "receipt_count": 7}
    with pytest.raises(PatchRunReceiptError, match="LEDGER_BACKUP_MANIFEST_INVALID"):
        verify_rank_ledger_backup(backup, tampered_manifest)

    payload = bytearray(backup.read_bytes())
    payload[-1] ^= 1
    backup.write_bytes(payload)
    with pytest.raises(PatchRunReceiptError, match="LEDGER_BACKUP_DIGEST_MISMATCH"):
        verify_rank_ledger_backup(backup, manifest)


@pytest.mark.parametrize("corruption", ["audit", "audit_digest", "receipt_count", "version"])
def test_logically_invalid_source_is_never_backed_up(
    tmp_path: Path,
    envelope: dict[str, object],
    corruption: str,
) -> None:
    ledger = _ingested_ledger(tmp_path, envelope)
    with sqlite3.connect(ledger) as connection:
        if corruption == "audit":
            connection.execute(
                "UPDATE promotions SET audit_json = ?",
                ('{"schema_version":"wrong"}',),
            )
        elif corruption == "audit_digest":
            audit = json.loads(connection.execute("SELECT audit_json FROM promotions").fetchone()[0])
            audit["evidence_sha256"] = "0" * 64
            connection.execute(
                "UPDATE promotions SET audit_json = ?",
                (json.dumps(audit, allow_nan=False, separators=(",", ":"), sort_keys=True),),
            )
        elif corruption == "receipt_count":
            connection.execute(
                "DELETE FROM consumed_receipts WHERE receipt_sha256 = "
                "(SELECT receipt_sha256 FROM consumed_receipts LIMIT 1)"
            )
        else:
            connection.execute("PRAGMA user_version = 2")
    backup = tmp_path / "backup.sqlite3"
    manifest_path = tmp_path / "backup.json"

    with pytest.raises(PatchRunReceiptError, match="LEDGER_BACKUP"):
        create_rank_ledger_backup(ledger, backup, manifest_path)

    assert not backup.exists()
    assert not manifest_path.exists()


def test_backup_paths_reject_symlinks_and_existing_targets(
    tmp_path: Path,
    envelope: dict[str, object],
) -> None:
    ledger = _ingested_ledger(tmp_path, envelope)
    source_link = tmp_path / "source-link.sqlite3"
    source_link.symlink_to(ledger)
    with pytest.raises(PatchRunReceiptError, match="REPLAY_LEDGER_INVALID"):
        create_rank_ledger_backup(source_link, tmp_path / "one.db", tmp_path / "one.json")

    backup = tmp_path / "backup.sqlite3"
    manifest_path = tmp_path / "backup.json"
    manifest = create_rank_ledger_backup(ledger, backup, manifest_path)
    with pytest.raises(PatchRunReceiptError, match="LEDGER_BACKUP_EXISTS"):
        create_rank_ledger_backup(ledger, backup, tmp_path / "second.json")

    restored = tmp_path / "restored.sqlite3"
    restored.write_text("do not replace", encoding="utf-8")
    with pytest.raises(PatchRunReceiptError, match="LEDGER_RESTORE_TARGET_EXISTS"):
        restore_rank_ledger_backup(backup, manifest, restored)
    assert restored.read_text(encoding="utf-8") == "do not replace"

    manifest_link = tmp_path / "manifest-link.json"
    manifest_link.symlink_to(manifest_path)
    with pytest.raises(PatchRunReceiptError, match="LEDGER_BACKUP_MANIFEST_INVALID"):
        load_rank_ledger_backup_manifest(manifest_link)


def test_manifest_loader_rejects_duplicate_and_unknown_fields(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":"one","schema_version":"two"}', encoding="utf-8")
    with pytest.raises(PatchRunReceiptError, match="LEDGER_BACKUP_MANIFEST_INVALID"):
        load_rank_ledger_backup_manifest(duplicate)

    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps({"unexpected": True}), encoding="utf-8")
    with pytest.raises(PatchRunReceiptError, match="LEDGER_BACKUP_MANIFEST_INVALID"):
        verify_rank_ledger_backup(tmp_path / "missing.db", load_rank_ledger_backup_manifest(unknown))


def test_backup_verify_restore_cli_workflow(
    tmp_path: Path,
    envelope: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ledger = _ingested_ledger(tmp_path, envelope)
    backup = tmp_path / "backup.sqlite3"
    manifest = tmp_path / "backup.json"
    restored = tmp_path / "restored.sqlite3"

    monkeypatch.setattr(
        "sys.argv",
        [
            "struct-xai",
            "backup-rank-ledger",
            "--ledger",
            str(ledger),
            "--backup",
            str(backup),
            "--manifest",
            str(manifest),
        ],
    )
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["promotion_count"] == 1

    monkeypatch.setattr(
        "sys.argv",
        [
            "struct-xai",
            "verify-rank-ledger-backup",
            "--backup",
            str(backup),
            "--manifest",
            str(manifest),
        ],
    )
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["verified"] is True

    monkeypatch.setattr(
        "sys.argv",
        [
            "struct-xai",
            "restore-rank-ledger-backup",
            "--backup",
            str(backup),
            "--manifest",
            str(manifest),
            "--ledger",
            str(restored),
        ],
    )
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["restored"] is True
