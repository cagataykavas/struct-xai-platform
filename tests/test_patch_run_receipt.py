import base64
import random
from dataclasses import replace

import pytest
import torch

from structxai.patch_dose_response import PatchSweepPoint, PatchSweepResult
from structxai.patch_run_receipt import (
    PatchRunReceiptError,
    PatchRunSigner,
    audit_authenticated_cross_seed_dose_response,
    verify_patch_run_receipt,
)

KEY = b"k" * 32
NOW = 1_800_000_000
MODEL_DIGEST = "a" * 64
TOKENIZER_DIGEST = "b" * 64


def sweep(endpoint: float = 0.7) -> PatchSweepResult:
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


def signer(run_id: str = "1" * 32) -> PatchRunSigner:
    return PatchRunSigner(
        "primary",
        KEY,
        clock=lambda: NOW,
        run_id_factory=lambda: run_id,
    )


def signed_run(seed: int, result: PatchSweepResult | None = None, **kwargs):
    active_result = result or sweep()
    return signer(f"{seed:032x}").run(
        lambda: active_result,
        seed=seed,
        model_artifact_sha256=kwargs.get("model_digest", MODEL_DIGEST),
        tokenizer_artifact_sha256=kwargs.get("tokenizer_digest", TOKENIZER_DIGEST),
    )


def test_seeded_boundary_authenticates_run_and_restores_caller_rng_state() -> None:
    random.seed(101)
    torch.manual_seed(202)
    python_state = random.getstate()
    torch_state = torch.get_rng_state().clone()

    result, receipt = signer().run(
        lambda: (random.random(), torch.rand(4), sweep())[-1],
        seed=42,
        model_artifact_sha256=MODEL_DIGEST,
        tokenizer_artifact_sha256=TOKENIZER_DIGEST,
    )

    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert receipt.seed == 42
    assert receipt.sweep_evidence_sha256 == result.evidence_sha256
    assert KEY.hex() not in repr(receipt)
    verify_patch_run_receipt(
        receipt,
        {"primary": KEY},
        expected_seed=42,
        expected_sweep_evidence_sha256=result.evidence_sha256,
        expected_model_artifact_sha256=MODEL_DIGEST,
        expected_tokenizer_artifact_sha256=TOKENIZER_DIGEST,
        now_epoch=NOW + 30,
    )


def test_same_seed_reproduces_initial_rng_digests() -> None:
    _, first = signed_run(42)
    _, second = signer("2" * 32).run(
        lambda: sweep(),
        seed=42,
        model_artifact_sha256=MODEL_DIGEST,
        tokenizer_artifact_sha256=TOKENIZER_DIGEST,
    )

    assert first.python_rng_before_sha256 == second.python_rng_before_sha256
    assert first.torch_rng_before_sha256 == second.torch_rng_before_sha256
    assert first.run_id != second.run_id
    assert first.signature != second.signature


def test_failed_callback_restores_rng_and_emits_no_receipt() -> None:
    random.seed(5)
    torch.manual_seed(6)
    python_state = random.getstate()
    torch_state = torch.get_rng_state().clone()

    def fail() -> PatchSweepResult:
        random.random()
        torch.rand(1)
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        signer().run(
            fail,
            seed=7,
            model_artifact_sha256=MODEL_DIGEST,
            tokenizer_artifact_sha256=TOKENIZER_DIGEST,
        )

    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("signature", "0" * 64, "RECEIPT_SIGNATURE_INVALID"),
        ("seed", 99, "RECEIPT_SIGNATURE_INVALID"),
        ("model_artifact_sha256", "0" * 64, "RECEIPT_SIGNATURE_INVALID"),
        ("run_id", "bad", "RECEIPT_SCHEMA_INVALID"),
    ],
)
def test_tampered_receipt_fails_closed(field: str, value: object, code: str) -> None:
    result, receipt = signed_run(42)
    tampered = replace(receipt, **{field: value})

    with pytest.raises(PatchRunReceiptError, match=code):
        verify_patch_run_receipt(
            tampered,
            {"primary": KEY},
            expected_seed=42,
            expected_sweep_evidence_sha256=result.evidence_sha256,
            expected_model_artifact_sha256=MODEL_DIGEST,
            expected_tokenizer_artifact_sha256=TOKENIZER_DIGEST,
            now_epoch=NOW,
        )


def test_freshness_and_expected_binding_are_enforced() -> None:
    result, receipt = signed_run(42)

    with pytest.raises(PatchRunReceiptError, match="RECEIPT_EXPIRED"):
        verify_patch_run_receipt(
            receipt,
            {"primary": KEY},
            expected_seed=42,
            expected_sweep_evidence_sha256=result.evidence_sha256,
            expected_model_artifact_sha256=MODEL_DIGEST,
            expected_tokenizer_artifact_sha256=TOKENIZER_DIGEST,
            now_epoch=NOW + 301,
        )
    with pytest.raises(PatchRunReceiptError, match="RECEIPT_BINDING_MISMATCH"):
        verify_patch_run_receipt(
            receipt,
            {"primary": KEY},
            expected_seed=43,
            expected_sweep_evidence_sha256=result.evidence_sha256,
            expected_model_artifact_sha256=MODEL_DIGEST,
            expected_tokenizer_artifact_sha256=TOKENIZER_DIGEST,
            now_epoch=NOW,
        )


def test_signer_loads_secret_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRUCTXAI_PATCH_RECEIPT_KEY_ID", "env-primary")
    monkeypatch.setenv(
        "STRUCTXAI_PATCH_RECEIPT_KEY_B64",
        base64.b64encode(KEY).decode("ascii"),
    )

    loaded = PatchRunSigner.from_env()
    result, receipt = loaded.run(
        lambda: sweep(),
        seed=8,
        model_artifact_sha256=MODEL_DIGEST,
        tokenizer_artifact_sha256=TOKENIZER_DIGEST,
    )

    verify_patch_run_receipt(
        receipt,
        {"env-primary": KEY},
        expected_seed=8,
        expected_sweep_evidence_sha256=result.evidence_sha256,
        expected_model_artifact_sha256=MODEL_DIGEST,
        expected_tokenizer_artifact_sha256=TOKENIZER_DIGEST,
        now_epoch=receipt.issued_at_epoch,
    )


def test_authenticated_cross_seed_audit_binds_receipts_and_artifacts() -> None:
    runs = {11: sweep(0.70), 22: sweep(0.72), 33: sweep(0.68)}
    receipts = {seed: signed_run(seed, result)[1] for seed, result in runs.items()}

    audit = audit_authenticated_cross_seed_dose_response(
        runs,
        receipts,
        {"primary": KEY},
        now_epoch=NOW,
    )
    reordered = audit_authenticated_cross_seed_dose_response(
        {33: runs[33], 11: runs[11], 22: runs[22]},
        {22: receipts[22], 33: receipts[33], 11: receipts[11]},
        {"primary": KEY},
        now_epoch=NOW,
    )

    assert audit.status == "accepted"
    assert audit.model_artifact_sha256 == MODEL_DIGEST
    assert audit.tokenizer_artifact_sha256 == TOKENIZER_DIGEST
    assert [run.seed for run in audit.seed_runs] == [11, 22, 33]
    assert audit.as_dict() == reordered.as_dict()


def test_authenticated_cross_seed_audit_rejects_missing_or_mixed_artifacts() -> None:
    runs = {11: sweep(0.70), 22: sweep(0.72), 33: sweep(0.68)}
    receipts = {seed: signed_run(seed, result)[1] for seed, result in runs.items()}

    with pytest.raises(PatchRunReceiptError, match="RUN_RECEIPT_SET_MISMATCH"):
        audit_authenticated_cross_seed_dose_response(
            runs,
            {11: receipts[11], 22: receipts[22]},
            {"primary": KEY},
            now_epoch=NOW,
        )

    receipts[33] = signed_run(33, runs[33], model_digest="9" * 64)[1]
    with pytest.raises(PatchRunReceiptError, match="CROSS_SEED_ARTIFACT_MISMATCH"):
        audit_authenticated_cross_seed_dose_response(
            runs,
            receipts,
            {"primary": KEY},
            now_epoch=NOW,
        )
