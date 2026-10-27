import base64
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import structxai.patch_run_receipt as receipt_module
from structxai.dataset_manifest import DatasetExample, build_dataset_manifest
from structxai.patch_dose_response import PatchSweepPoint, PatchSweepResult
from structxai.patch_run_receipt import (
    PatchRunReceiptError,
    PatchRunSigner,
    audit_authenticated_cross_seed_dose_response,
    verify_patch_run_receipt,
)
from structxai.sampler_provenance import build_distributed_sampler_plan
from structxai.worker_seeding import build_worker_seed_plan

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


def numpy_states_equal(left: tuple, right: tuple) -> bool:
    return left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]


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
    np.random.seed(303)
    torch.manual_seed(202)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()

    result, receipt = signer().run(
        lambda: (random.random(), np.random.random(4), torch.rand(4), sweep())[-1],
        seed=42,
        model_artifact_sha256=MODEL_DIGEST,
        tokenizer_artifact_sha256=TOKENIZER_DIGEST,
    )

    assert random.getstate() == python_state
    assert numpy_states_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert receipt.seed == 42
    assert receipt.cuda_device_count == (torch.cuda.device_count() if torch.cuda.is_available() else 0)
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
    assert first.numpy_rng_before_sha256 == second.numpy_rng_before_sha256
    assert first.torch_rng_before_sha256 == second.torch_rng_before_sha256
    assert first.cuda_rng_before_sha256 == second.cuda_rng_before_sha256
    assert first.run_id != second.run_id
    assert first.signature != second.signature


def test_failed_callback_restores_rng_and_emits_no_receipt() -> None:
    random.seed(5)
    np.random.seed(7)
    torch.manual_seed(6)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()

    def fail() -> PatchSweepResult:
        random.random()
        np.random.random()
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
    assert numpy_states_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_cuda_rng_state_is_seeded_hashed_and_restored_without_cuda_hardware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_states = [
        torch.tensor([1, 2, 3], dtype=torch.uint8),
        torch.tensor([4, 5, 6], dtype=torch.uint8),
    ]
    active_states = [state.clone() for state in original_states]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [state.clone() for state in active_states])

    def seed_all(seed: int) -> None:
        active_states[:] = [
            torch.tensor([(seed + device) % 256, 10, 20], dtype=torch.uint8) for device in range(2)
        ]

    def restore_all(states: list[torch.Tensor]) -> None:
        active_states[:] = [state.clone() for state in states]

    monkeypatch.setattr(torch.cuda, "manual_seed_all", seed_all)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restore_all)

    def consume_cuda_state() -> PatchSweepResult:
        active_states[0][1] = 99
        active_states[1][1] = 98
        return sweep()

    _, receipt = signer().run(
        consume_cuda_state,
        seed=42,
        model_artifact_sha256=MODEL_DIGEST,
        tokenizer_artifact_sha256=TOKENIZER_DIGEST,
    )

    assert receipt.cuda_device_count == 2
    assert receipt.cuda_rng_before_sha256 != receipt.cuda_rng_after_sha256
    assert all(torch.equal(actual, expected) for actual, expected in zip(active_states, original_states))


def test_rng_boundary_fails_closed_when_process_lock_is_busy() -> None:
    locked = receipt_module._RNG_LOCK.acquire(timeout=0)
    assert locked
    try:
        blocked = PatchRunSigner(
            "primary",
            KEY,
            clock=lambda: NOW,
            run_id_factory=lambda: "1" * 32,
            rng_lock_timeout_seconds=0.01,
        )
        with pytest.raises(PatchRunReceiptError, match="RNG_BOUNDARY_BUSY"):
            blocked.run(
                lambda: sweep(),
                seed=42,
                model_artifact_sha256=MODEL_DIGEST,
                tokenizer_artifact_sha256=TOKENIZER_DIGEST,
            )
    finally:
        receipt_module._RNG_LOCK.release()


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf"), True, "1"])
def test_rng_lock_policy_rejects_invalid_timeout(timeout: object) -> None:
    with pytest.raises(PatchRunReceiptError, match="RNG_LOCK_POLICY_INVALID"):
        PatchRunSigner("primary", KEY, rng_lock_timeout_seconds=timeout)  # type: ignore[arg-type]


def test_run_from_artifacts_computes_and_signs_manifests(tmp_path: Path) -> None:
    model = tmp_path / "model"
    tokenizer = tmp_path / "tokenizer"
    model.mkdir()
    tokenizer.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")
    (tokenizer / "tokenizer.json").write_text('{"vocab":[]}', encoding="utf-8")

    result, receipt, model_manifest, tokenizer_manifest = signer().run_from_artifacts(
        lambda: sweep(),
        seed=42,
        model_path=model,
        tokenizer_path=tokenizer,
    )

    assert receipt.model_artifact_sha256 == model_manifest.evidence_sha256
    assert receipt.tokenizer_artifact_sha256 == tokenizer_manifest.evidence_sha256
    verify_patch_run_receipt(
        receipt,
        {"primary": KEY},
        expected_seed=42,
        expected_sweep_evidence_sha256=result.evidence_sha256,
        expected_model_artifact_sha256=model_manifest.evidence_sha256,
        expected_tokenizer_artifact_sha256=tokenizer_manifest.evidence_sha256,
        now_epoch=NOW,
    )


@pytest.mark.parametrize("target", ["model", "tokenizer"])
def test_run_from_artifacts_rejects_mutation_during_callback(tmp_path: Path, target: str) -> None:
    model = tmp_path / "model"
    tokenizer = tmp_path / "tokenizer"
    model.mkdir()
    tokenizer.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")
    (tokenizer / "tokenizer.json").write_bytes(b"tokens")

    def mutate() -> PatchSweepResult:
        destination = model / "model.safetensors" if target == "model" else tokenizer / "tokenizer.json"
        destination.write_bytes(b"changed")
        return sweep()

    expected = f"{target.upper()}_ARTIFACT_CHANGED_DURING_RUN"
    with pytest.raises(PatchRunReceiptError, match=expected):
        signer().run_from_artifacts(
            mutate,
            seed=42,
            model_path=model,
            tokenizer_path=tokenizer,
        )


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


def test_authenticated_cross_seed_audit_rejects_mixed_cuda_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = {11: sweep(0.70), 22: sweep(0.72)}
    cpu_receipt = signed_run(11, runs[11])[1]
    active_state = [torch.tensor([3, 4, 5], dtype=torch.uint8)]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [active_state[0].clone()])
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda seed: active_state.__setitem__(0, torch.tensor([seed % 256, 4, 5], dtype=torch.uint8)),
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda states: active_state.__setitem__(0, states[0].clone()),
    )
    gpu_receipt = signed_run(22, runs[22])[1]

    with pytest.raises(PatchRunReceiptError, match="CROSS_SEED_RNG_BACKEND_MISMATCH"):
        audit_authenticated_cross_seed_dose_response(
            runs,
            {11: cpu_receipt, 22: gpu_receipt},
            {"primary": KEY},
            now_epoch=NOW,
        )


def test_worker_plan_is_applied_and_authenticated() -> None:
    observed: dict[str, float] = {}

    def callback(initializer) -> PatchSweepResult:
        initializer(1)
        observed["python"] = random.random()
        observed["numpy"] = float(np.random.random())
        observed["torch"] = float(torch.rand(1).item())
        return sweep()

    result, receipt = signer().run_with_worker_plan(
        callback,
        seed=42,
        model_artifact_sha256=MODEL_DIGEST,
        tokenizer_artifact_sha256=TOKENIZER_DIGEST,
        world_size=2,
        rank=1,
        workers_per_rank=2,
        epoch=3,
    )
    plan = build_worker_seed_plan(42, world_size=2, rank=1, workers_per_rank=2, epoch=3)

    assert set(observed) == {"python", "numpy", "torch"}
    assert receipt.worker_seed_manifest_sha256 == plan.manifest.evidence_sha256
    assert (
        receipt.worker_world_size,
        receipt.worker_rank,
        receipt.workers_per_rank,
        receipt.worker_epoch,
    ) == (2, 1, 2, 3)
    assert all(str(entry.seed) not in repr(receipt) for entry in plan.initializer.entries)
    verify_patch_run_receipt(
        receipt,
        {"primary": KEY},
        expected_seed=42,
        expected_sweep_evidence_sha256=result.evidence_sha256,
        expected_model_artifact_sha256=MODEL_DIGEST,
        expected_tokenizer_artifact_sha256=TOKENIZER_DIGEST,
        expected_worker_seed_manifest_sha256=plan.manifest.evidence_sha256,
        expected_worker_topology=(2, 1, 2, 3),
        now_epoch=NOW,
    )


def test_input_plan_drives_dataloader_and_is_authenticated() -> None:
    observed: list[int] = []

    def callback(_initializer, sampler) -> PatchSweepResult:
        loader = DataLoader(
            list(range(9)),
            batch_size=2,
            sampler=sampler,
            num_workers=0,
        )
        observed.extend(int(value) for batch in loader for value in batch)
        return sweep()

    result, receipt = signer().run_with_input_plan(
        callback,
        seed=42,
        model_artifact_sha256=MODEL_DIGEST,
        tokenizer_artifact_sha256=TOKENIZER_DIGEST,
        world_size=2,
        rank=1,
        workers_per_rank=2,
        dataset_size=9,
        epoch=3,
    )
    plan = build_distributed_sampler_plan(
        42,
        world_size=2,
        rank=1,
        dataset_size=9,
        epoch=3,
    )

    assert observed == list(plan.sampler.indices)
    assert receipt.sampler_bound is True
    assert receipt.sampler_manifest_sha256 == plan.manifest.evidence_sha256
    assert "indices" not in repr(receipt)
    verify_patch_run_receipt(
        receipt,
        {"primary": KEY},
        expected_seed=42,
        expected_sweep_evidence_sha256=result.evidence_sha256,
        expected_model_artifact_sha256=MODEL_DIGEST,
        expected_tokenizer_artifact_sha256=TOKENIZER_DIGEST,
        expected_sampler_manifest_sha256=plan.manifest.evidence_sha256,
        now_epoch=NOW,
    )


def test_acknowledged_input_plan_requires_every_worker_and_binds_manifest() -> None:
    def callback(initializer, _sampler) -> PatchSweepResult:
        initializer(0)
        initializer(1)
        return sweep()

    result, receipt, acknowledgment_manifest = signer().run_with_acknowledged_input_plan(
        callback,
        seed=42,
        model_artifact_sha256=MODEL_DIGEST,
        tokenizer_artifact_sha256=TOKENIZER_DIGEST,
        world_size=1,
        rank=0,
        workers_per_rank=2,
        dataset_size=8,
        epoch=3,
    )

    assert receipt.worker_acknowledgment_bound is True
    assert receipt.worker_acknowledgment_manifest_sha256 == acknowledgment_manifest.evidence_sha256
    assert [item.worker_id for item in acknowledgment_manifest.acknowledgments] == [0, 1]
    verify_patch_run_receipt(
        receipt,
        {"primary": KEY},
        expected_seed=42,
        expected_sweep_evidence_sha256=result.evidence_sha256,
        expected_model_artifact_sha256=MODEL_DIGEST,
        expected_tokenizer_artifact_sha256=TOKENIZER_DIGEST,
        expected_worker_acknowledgment_manifest_sha256=(acknowledgment_manifest.evidence_sha256),
        now_epoch=NOW,
    )


def test_acknowledged_input_plan_rejects_ignored_initializer() -> None:
    with pytest.raises(PatchRunReceiptError, match="WORKER_ACK_MISSING"):
        signer().run_with_acknowledged_input_plan(
            lambda _initializer, _sampler: sweep(),
            seed=42,
            model_artifact_sha256=MODEL_DIGEST,
            tokenizer_artifact_sha256=TOKENIZER_DIGEST,
            world_size=1,
            rank=0,
            workers_per_rank=2,
            dataset_size=8,
        )


def test_worker_acknowledgment_manifest_tampering_breaks_signature() -> None:
    _, receipt, _manifest = signer().run_with_acknowledged_input_plan(
        lambda initializer, _sampler: (initializer(0), sweep())[1],
        seed=42,
        model_artifact_sha256=MODEL_DIGEST,
        tokenizer_artifact_sha256=TOKENIZER_DIGEST,
        world_size=1,
        rank=0,
        workers_per_rank=1,
        dataset_size=8,
    )

    with pytest.raises(PatchRunReceiptError, match="RECEIPT_SIGNATURE_INVALID"):
        verify_patch_run_receipt(
            replace(receipt, worker_acknowledgment_manifest_sha256="9" * 64),
            {"primary": KEY},
            expected_seed=42,
            expected_sweep_evidence_sha256=receipt.sweep_evidence_sha256,
            expected_model_artifact_sha256=MODEL_DIGEST,
            expected_tokenizer_artifact_sha256=TOKENIZER_DIGEST,
            now_epoch=NOW,
        )


def test_input_plan_binds_exact_dataset_contents_to_signed_receipt() -> None:
    values = [
        DatasetExample("example-1", b"first payload"),
        DatasetExample("example-2", b"second payload"),
        DatasetExample("example-3", b"third payload"),
    ]
    dataset_manifest = build_dataset_manifest(values)

    result, receipt = signer().run_with_input_plan(
        lambda _initializer, _sampler: sweep(),
        seed=42,
        model_artifact_sha256=MODEL_DIGEST,
        tokenizer_artifact_sha256=TOKENIZER_DIGEST,
        world_size=1,
        rank=0,
        workers_per_rank=1,
        dataset_size=3,
        dataset_examples=values,
    )
    plan = build_distributed_sampler_plan(
        42,
        world_size=1,
        rank=0,
        dataset_size=3,
        dataset_manifest=dataset_manifest,
    )

    assert receipt.dataset_bound is True
    assert receipt.dataset_manifest_sha256 == dataset_manifest.evidence_sha256
    assert receipt.sampler_manifest_sha256 == plan.manifest.evidence_sha256
    assert "first payload" not in repr(receipt)
    verify_patch_run_receipt(
        receipt,
        {"primary": KEY},
        expected_seed=42,
        expected_sweep_evidence_sha256=result.evidence_sha256,
        expected_model_artifact_sha256=MODEL_DIGEST,
        expected_tokenizer_artifact_sha256=TOKENIZER_DIGEST,
        expected_sampler_manifest_sha256=plan.manifest.evidence_sha256,
        expected_dataset_manifest_sha256=dataset_manifest.evidence_sha256,
        now_epoch=NOW,
    )


def test_dataset_mutation_during_callback_fails_closed() -> None:
    values = [
        DatasetExample("example-1", b"first payload"),
        DatasetExample("example-2", b"second payload"),
    ]

    def mutate(_initializer, _sampler) -> PatchSweepResult:
        values[0] = DatasetExample("example-1", b"changed payload")
        return sweep()

    with pytest.raises(PatchRunReceiptError, match="DATASET_CHANGED_DURING_RUN"):
        signer().run_with_input_plan(
            mutate,
            seed=42,
            model_artifact_sha256=MODEL_DIGEST,
            tokenizer_artifact_sha256=TOKENIZER_DIGEST,
            world_size=1,
            rank=0,
            workers_per_rank=1,
            dataset_size=2,
            dataset_examples=values,
        )


def test_input_plan_mutation_during_callback_fails_closed() -> None:
    def mutate(_initializer, sampler) -> PatchSweepResult:
        object.__setattr__(sampler, "indices", tuple(reversed(sampler.indices)))
        return sweep()

    with pytest.raises(
        PatchRunReceiptError,
        match="SAMPLER_PLAN_CHANGED_DURING_RUN",
    ):
        signer().run_with_input_plan(
            mutate,
            seed=42,
            model_artifact_sha256=MODEL_DIGEST,
            tokenizer_artifact_sha256=TOKENIZER_DIGEST,
            world_size=2,
            rank=0,
            workers_per_rank=2,
            dataset_size=8,
            epoch=1,
        )


def test_sampler_manifest_tampering_breaks_signature() -> None:
    _, receipt = signer().run_with_input_plan(
        lambda _initializer, _sampler: sweep(),
        seed=42,
        model_artifact_sha256=MODEL_DIGEST,
        tokenizer_artifact_sha256=TOKENIZER_DIGEST,
        world_size=1,
        rank=0,
        workers_per_rank=1,
        dataset_size=8,
    )

    with pytest.raises(PatchRunReceiptError, match="RECEIPT_SIGNATURE_INVALID"):
        verify_patch_run_receipt(
            replace(receipt, sampler_manifest_sha256="9" * 64),
            {"primary": KEY},
            expected_seed=42,
            expected_sweep_evidence_sha256=receipt.sweep_evidence_sha256,
            expected_model_artifact_sha256=MODEL_DIGEST,
            expected_tokenizer_artifact_sha256=TOKENIZER_DIGEST,
            now_epoch=NOW,
        )


def test_dataset_manifest_tampering_breaks_signature() -> None:
    values = [
        DatasetExample("example-1", b"first payload"),
        DatasetExample("example-2", b"second payload"),
    ]
    _, receipt = signer().run_with_input_plan(
        lambda _initializer, _sampler: sweep(),
        seed=42,
        model_artifact_sha256=MODEL_DIGEST,
        tokenizer_artifact_sha256=TOKENIZER_DIGEST,
        world_size=1,
        rank=0,
        workers_per_rank=1,
        dataset_size=2,
        dataset_examples=values,
    )

    with pytest.raises(PatchRunReceiptError, match="RECEIPT_SIGNATURE_INVALID"):
        verify_patch_run_receipt(
            replace(receipt, dataset_manifest_sha256="9" * 64),
            {"primary": KEY},
            expected_seed=42,
            expected_sweep_evidence_sha256=receipt.sweep_evidence_sha256,
            expected_model_artifact_sha256=MODEL_DIGEST,
            expected_tokenizer_artifact_sha256=TOKENIZER_DIGEST,
            now_epoch=NOW,
        )


def test_unbound_run_cannot_satisfy_expected_sampler_manifest() -> None:
    result, receipt = signer().run(
        sweep,
        seed=42,
        model_artifact_sha256=MODEL_DIGEST,
        tokenizer_artifact_sha256=TOKENIZER_DIGEST,
    )

    with pytest.raises(PatchRunReceiptError, match="RECEIPT_BINDING_MISMATCH"):
        verify_patch_run_receipt(
            receipt,
            {"primary": KEY},
            expected_seed=42,
            expected_sweep_evidence_sha256=result.evidence_sha256,
            expected_model_artifact_sha256=MODEL_DIGEST,
            expected_tokenizer_artifact_sha256=TOKENIZER_DIGEST,
            expected_sampler_manifest_sha256=receipt.sampler_manifest_sha256,
            now_epoch=NOW,
        )


def test_unbound_run_cannot_satisfy_expected_dataset_manifest() -> None:
    result, receipt = signer().run(
        sweep,
        seed=42,
        model_artifact_sha256=MODEL_DIGEST,
        tokenizer_artifact_sha256=TOKENIZER_DIGEST,
    )

    with pytest.raises(PatchRunReceiptError, match="RECEIPT_BINDING_MISMATCH"):
        verify_patch_run_receipt(
            receipt,
            {"primary": KEY},
            expected_seed=42,
            expected_sweep_evidence_sha256=result.evidence_sha256,
            expected_model_artifact_sha256=MODEL_DIGEST,
            expected_tokenizer_artifact_sha256=TOKENIZER_DIGEST,
            expected_dataset_manifest_sha256=receipt.dataset_manifest_sha256,
            now_epoch=NOW,
        )


def test_worker_manifest_tampering_breaks_signature() -> None:
    _, receipt = signer().run_with_worker_plan(
        lambda _initializer: sweep(),
        seed=42,
        model_artifact_sha256=MODEL_DIGEST,
        tokenizer_artifact_sha256=TOKENIZER_DIGEST,
        world_size=1,
        rank=0,
        workers_per_rank=2,
    )

    with pytest.raises(PatchRunReceiptError, match="RECEIPT_SIGNATURE_INVALID"):
        verify_patch_run_receipt(
            replace(receipt, worker_seed_manifest_sha256="9" * 64),
            {"primary": KEY},
            expected_seed=42,
            expected_sweep_evidence_sha256=receipt.sweep_evidence_sha256,
            expected_model_artifact_sha256=MODEL_DIGEST,
            expected_tokenizer_artifact_sha256=TOKENIZER_DIGEST,
            now_epoch=NOW,
        )


def test_authenticated_cross_seed_audit_rejects_mixed_worker_topology() -> None:
    runs = {11: sweep(0.70), 22: sweep(0.72)}
    _, first = signer("1" * 32).run_with_worker_plan(
        lambda _initializer: runs[11],
        seed=11,
        model_artifact_sha256=MODEL_DIGEST,
        tokenizer_artifact_sha256=TOKENIZER_DIGEST,
        world_size=1,
        rank=0,
        workers_per_rank=2,
    )
    _, second = signer("2" * 32).run_with_worker_plan(
        lambda _initializer: runs[22],
        seed=22,
        model_artifact_sha256=MODEL_DIGEST,
        tokenizer_artifact_sha256=TOKENIZER_DIGEST,
        world_size=1,
        rank=0,
        workers_per_rank=3,
    )

    with pytest.raises(
        PatchRunReceiptError,
        match="CROSS_SEED_WORKER_TOPOLOGY_MISMATCH",
    ):
        audit_authenticated_cross_seed_dose_response(
            runs,
            {11: first, 22: second},
            {"primary": KEY},
            now_epoch=NOW,
        )


def test_authenticated_cross_seed_audit_rejects_mixed_sampler_boundary() -> None:
    runs = {11: sweep(0.70), 22: sweep(0.72)}
    _, worker_only = signer("1" * 32).run_with_worker_plan(
        lambda _initializer: runs[11],
        seed=11,
        model_artifact_sha256=MODEL_DIGEST,
        tokenizer_artifact_sha256=TOKENIZER_DIGEST,
        world_size=1,
        rank=0,
        workers_per_rank=1,
    )
    _, input_bound = signer("2" * 32).run_with_input_plan(
        lambda _initializer, _sampler: runs[22],
        seed=22,
        model_artifact_sha256=MODEL_DIGEST,
        tokenizer_artifact_sha256=TOKENIZER_DIGEST,
        world_size=1,
        rank=0,
        workers_per_rank=1,
        dataset_size=8,
    )

    with pytest.raises(
        PatchRunReceiptError,
        match="CROSS_SEED_SAMPLER_BOUNDARY_MISMATCH",
    ):
        audit_authenticated_cross_seed_dose_response(
            runs,
            {11: worker_only, 22: input_bound},
            {"primary": KEY},
            now_epoch=NOW,
        )


def test_authenticated_cross_seed_audit_rejects_mixed_worker_ack_boundary() -> None:
    runs = {11: sweep(0.70), 22: sweep(0.72)}
    _, ordinary = signer("1" * 32).run_with_input_plan(
        lambda _initializer, _sampler: runs[11],
        seed=11,
        model_artifact_sha256=MODEL_DIGEST,
        tokenizer_artifact_sha256=TOKENIZER_DIGEST,
        world_size=1,
        rank=0,
        workers_per_rank=1,
        dataset_size=8,
    )
    _, acknowledged, _manifest = signer("2" * 32).run_with_acknowledged_input_plan(
        lambda initializer, _sampler: (initializer(0), runs[22])[1],
        seed=22,
        model_artifact_sha256=MODEL_DIGEST,
        tokenizer_artifact_sha256=TOKENIZER_DIGEST,
        world_size=1,
        rank=0,
        workers_per_rank=1,
        dataset_size=8,
    )

    with pytest.raises(
        PatchRunReceiptError,
        match="CROSS_SEED_WORKER_ACK_BOUNDARY_MISMATCH",
    ):
        audit_authenticated_cross_seed_dose_response(
            runs,
            {11: ordinary, 22: acknowledged},
            {"primary": KEY},
            now_epoch=NOW,
        )


def test_authenticated_cross_seed_audit_rejects_different_dataset_contents() -> None:
    runs = {11: sweep(0.70), 22: sweep(0.72)}
    first_dataset = [
        DatasetExample("example-1", b"first payload"),
        DatasetExample("example-2", b"second payload"),
    ]
    second_dataset = [
        DatasetExample("example-1", b"changed payload"),
        DatasetExample("example-2", b"second payload"),
    ]
    _, first = signer("1" * 32).run_with_input_plan(
        lambda _initializer, _sampler: runs[11],
        seed=11,
        model_artifact_sha256=MODEL_DIGEST,
        tokenizer_artifact_sha256=TOKENIZER_DIGEST,
        world_size=1,
        rank=0,
        workers_per_rank=1,
        dataset_size=2,
        dataset_examples=first_dataset,
    )
    _, second = signer("2" * 32).run_with_input_plan(
        lambda _initializer, _sampler: runs[22],
        seed=22,
        model_artifact_sha256=MODEL_DIGEST,
        tokenizer_artifact_sha256=TOKENIZER_DIGEST,
        world_size=1,
        rank=0,
        workers_per_rank=1,
        dataset_size=2,
        dataset_examples=second_dataset,
    )

    with pytest.raises(PatchRunReceiptError, match="CROSS_SEED_DATASET_MISMATCH"):
        audit_authenticated_cross_seed_dose_response(
            runs,
            {11: first, 22: second},
            {"primary": KEY},
            now_epoch=NOW,
        )
