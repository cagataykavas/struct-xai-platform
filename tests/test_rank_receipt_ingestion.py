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
