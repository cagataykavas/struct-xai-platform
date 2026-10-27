from __future__ import annotations

import multiprocessing
import pickle

import pytest

from structxai.worker_acknowledgment import (
    WorkerAcknowledgmentError,
    WorkerAcknowledgmentSession,
    verify_worker_acknowledgment_manifest,
)
from structxai.worker_seeding import build_worker_seed_plan


def _initialize_worker(initializer: object, worker_id: int) -> None:
    initializer(worker_id)  # type: ignore[operator]


def test_initializer_records_complete_cross_process_manifest() -> None:
    plan = build_worker_seed_plan(42, world_size=2, rank=1, workers_per_rank=2, epoch=3)

    with WorkerAcknowledgmentSession(plan) as session:
        raw_challenge = session.initializer.challenge
        restored = pickle.loads(pickle.dumps(session.initializer))
        context = multiprocessing.get_context("fork")
        processes = [
            context.Process(target=_initialize_worker, args=(restored, worker_id)) for worker_id in range(2)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        manifest = session.finalize()

    assert verify_worker_acknowledgment_manifest(manifest, plan) is True
    assert [item.worker_id for item in manifest.acknowledgments] == [0, 1]
    assert manifest.worker_seed_manifest_sha256 == plan.manifest.evidence_sha256
    assert all(str(entry.seed) not in repr(manifest) for entry in plan.initializer.entries)
    assert raw_challenge not in repr(manifest)


def test_missing_or_duplicate_worker_acknowledgment_fails_closed() -> None:
    plan = build_worker_seed_plan(7, world_size=1, rank=0, workers_per_rank=2)

    with WorkerAcknowledgmentSession(plan) as session:
        session.initializer(0)
        with pytest.raises(WorkerAcknowledgmentError, match="WORKER_ACK_DUPLICATE"):
            session.initializer(0)
        with pytest.raises(WorkerAcknowledgmentError, match="WORKER_ACK_MISSING"):
            session.finalize()


def test_unexpected_acknowledgment_file_fails_closed(tmp_path) -> None:
    plan = build_worker_seed_plan(7, world_size=1, rank=0, workers_per_rank=1)

    with WorkerAcknowledgmentSession(plan) as session:
        session.initializer(0)
        unexpected = tmp_path / "unexpected.json"
        unexpected.write_text("{}", encoding="utf-8")
        target = session.initializer.directory + "/unexpected.json"
        with open(target, "wb") as handle:
            handle.write(unexpected.read_bytes())
        with pytest.raises(WorkerAcknowledgmentError, match="WORKER_ACK_UNEXPECTED"):
            session.finalize()
