from __future__ import annotations

import pickle
import random

import numpy as np
import pytest
import torch

from structxai.worker_seeding import (
    MAX_TOTAL_WORKERS,
    WorkerSeedError,
    build_worker_seed_plan,
    verify_worker_seed_plan,
)


def test_global_manifest_is_shared_while_local_initializers_are_rank_scoped() -> None:
    first = build_worker_seed_plan(
        42,
        world_size=2,
        rank=0,
        workers_per_rank=3,
        epoch=7,
    )
    second = build_worker_seed_plan(
        42,
        world_size=2,
        rank=1,
        workers_per_rank=3,
        epoch=7,
    )

    assert first.manifest == second.manifest
    assert first.initializer.entries != second.initializer.entries
    assert len(first.manifest.commitments) == 6
    assert len({item.seed_sha256 for item in first.manifest.commitments}) == 6
    assert verify_worker_seed_plan(first, 42) is True
    assert verify_worker_seed_plan(first, 43) is False


def test_epoch_and_root_seed_change_manifest_identity() -> None:
    baseline = build_worker_seed_plan(11, world_size=1, rank=0, workers_per_rank=2, epoch=0)
    next_epoch = build_worker_seed_plan(11, world_size=1, rank=0, workers_per_rank=2, epoch=1)
    next_seed = build_worker_seed_plan(12, world_size=1, rank=0, workers_per_rank=2, epoch=0)

    assert baseline.manifest.evidence_sha256 != next_epoch.manifest.evidence_sha256
    assert baseline.manifest.evidence_sha256 != next_seed.manifest.evidence_sha256


def test_initializer_seeds_python_numpy_and_torch() -> None:
    plan = build_worker_seed_plan(99, world_size=1, rank=0, workers_per_rank=2, epoch=3)
    worker_seed = plan.initializer.entries[1].seed

    plan.initializer(1)
    actual = (random.random(), float(np.random.random()), float(torch.rand(1).item()))

    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))
    torch.manual_seed(worker_seed)
    expected = (random.random(), float(np.random.random()), float(torch.rand(1).item()))
    assert actual == expected


def test_initializer_is_picklable_and_rejects_unknown_worker() -> None:
    plan = build_worker_seed_plan(7, world_size=2, rank=1, workers_per_rank=2, epoch=0)
    restored = pickle.loads(pickle.dumps(plan.initializer))

    assert restored == plan.initializer
    with pytest.raises(WorkerSeedError, match="WORKER_ID_OUT_OF_RANGE"):
        restored(2)


def test_manifest_does_not_publish_raw_worker_seeds() -> None:
    plan = build_worker_seed_plan(123, world_size=2, rank=0, workers_per_rank=2, epoch=0)
    serialized = repr(plan.manifest.as_dict())

    for entry in plan.initializer.entries:
        assert str(entry.seed) not in serialized


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"world_size": 0, "rank": 0, "workers_per_rank": 1}, "WORLD_SIZE_INVALID"),
        ({"world_size": 2, "rank": 2, "workers_per_rank": 1}, "RANK_INVALID"),
        ({"world_size": 1, "rank": 0, "workers_per_rank": 0}, "WORKER_COUNT_INVALID"),
        (
            {
                "world_size": MAX_TOTAL_WORKERS,
                "rank": 0,
                "workers_per_rank": 2,
            },
            "WORLD_SIZE_INVALID",
        ),
        ({"world_size": 1, "rank": 0, "workers_per_rank": 1, "epoch": -1}, "EPOCH_INVALID"),
    ],
)
def test_invalid_or_unbounded_topology_fails_closed(kwargs: dict, code: str) -> None:
    with pytest.raises(WorkerSeedError, match=code):
        build_worker_seed_plan(1, **kwargs)
