from __future__ import annotations

import pickle

import pytest
from torch.utils.data import DataLoader

from structxai.dataset_manifest import DatasetExample, build_dataset_manifest
from structxai.sampler_provenance import (
    MAX_DATASET_SIZE,
    SamplerPlanError,
    build_distributed_sampler_plan,
    verify_distributed_sampler_plan,
)


def test_all_ranks_share_manifest_and_partition_padded_order() -> None:
    plans = [
        build_distributed_sampler_plan(
            42,
            world_size=3,
            rank=rank,
            dataset_size=8,
            epoch=4,
        )
        for rank in range(3)
    ]

    assert plans[0].manifest == plans[1].manifest == plans[2].manifest
    assert all(len(plan.sampler) == 3 for plan in plans)
    assert sum(len(plan.sampler) for plan in plans) == 9
    assert all(verify_distributed_sampler_plan(plan, 42) for plan in plans)
    assert verify_distributed_sampler_plan(plans[0], 43) is False


def test_sampler_is_picklable_and_drives_real_dataloader_order() -> None:
    plan = build_distributed_sampler_plan(
        7,
        world_size=2,
        rank=1,
        dataset_size=9,
        epoch=2,
        shuffle=True,
        drop_last=False,
    )
    restored = pickle.loads(pickle.dumps(plan.sampler))
    loader = DataLoader(list(range(9)), batch_size=2, sampler=restored)

    observed = [int(value) for batch in loader for value in batch]
    assert observed == list(plan.sampler.indices)


def test_epoch_seed_and_policy_change_evidence_identity() -> None:
    baseline = build_distributed_sampler_plan(11, world_size=2, rank=0, dataset_size=10, epoch=0)
    next_epoch = build_distributed_sampler_plan(11, world_size=2, rank=0, dataset_size=10, epoch=1)
    next_seed = build_distributed_sampler_plan(12, world_size=2, rank=0, dataset_size=10, epoch=0)
    no_shuffle = build_distributed_sampler_plan(
        11, world_size=2, rank=0, dataset_size=10, epoch=0, shuffle=False
    )

    identities = {plan.manifest.evidence_sha256 for plan in (baseline, next_epoch, next_seed, no_shuffle)}
    assert len(identities) == 4


def test_non_shuffled_drop_last_partition_is_exact_and_disjoint() -> None:
    plans = [
        build_distributed_sampler_plan(
            5,
            world_size=3,
            rank=rank,
            dataset_size=8,
            shuffle=False,
            drop_last=True,
        )
        for rank in range(3)
    ]

    assert [plan.sampler.indices for plan in plans] == [(0, 3), (1, 4), (2, 5)]
    assert plans[0].manifest.total_size == 6
    assert plans[0].manifest.samples_per_rank == 2


def test_manifest_does_not_publish_raw_sample_order() -> None:
    plan = build_distributed_sampler_plan(13, world_size=2, rank=0, dataset_size=100, epoch=3)

    assert "indices" not in repr(plan.manifest.as_dict())
    assert plan.manifest.rank_orders[0].sample_count == 50
    assert len(plan.manifest.rank_orders[0].order_sha256) == 64


def test_content_manifest_changes_sampler_evidence_without_changing_order() -> None:
    first_dataset = build_dataset_manifest([DatasetExample("a", b"first"), DatasetExample("b", b"second")])
    second_dataset = build_dataset_manifest([DatasetExample("a", b"changed"), DatasetExample("b", b"second")])

    first = build_distributed_sampler_plan(
        13,
        world_size=1,
        rank=0,
        dataset_size=2,
        dataset_manifest=first_dataset,
    )
    second = build_distributed_sampler_plan(
        13,
        world_size=1,
        rank=0,
        dataset_size=2,
        dataset_manifest=second_dataset,
    )

    assert first.sampler == second.sampler
    assert first.manifest.dataset_bound is True
    assert first.manifest.dataset_manifest_sha256 == first_dataset.evidence_sha256
    assert first.manifest.evidence_sha256 != second.manifest.evidence_sha256
    assert "first" not in repr(first.manifest.as_dict())


def test_sampler_rejects_dataset_manifest_size_mismatch() -> None:
    dataset = build_dataset_manifest([DatasetExample("a", b"first")])

    with pytest.raises(SamplerPlanError, match="SAMPLER_DATASET_SIZE_MISMATCH"):
        build_distributed_sampler_plan(
            13,
            world_size=1,
            rank=0,
            dataset_size=2,
            dataset_manifest=dataset,
        )


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"world_size": 0, "rank": 0, "dataset_size": 2}, "SAMPLER_WORLD_SIZE_INVALID"),
        ({"world_size": 2, "rank": 2, "dataset_size": 2}, "SAMPLER_RANK_INVALID"),
        (
            {"world_size": 1, "rank": 0, "dataset_size": MAX_DATASET_SIZE + 1},
            "SAMPLER_DATASET_SIZE_INVALID",
        ),
        (
            {"world_size": 1, "rank": 0, "dataset_size": 2, "shuffle": 1},
            "SAMPLER_SHUFFLE_INVALID",
        ),
        (
            {"world_size": 1, "rank": 0, "dataset_size": 2, "drop_last": 0},
            "SAMPLER_DROP_LAST_INVALID",
        ),
    ],
)
def test_invalid_sampler_policy_fails_closed(kwargs: dict, code: str) -> None:
    with pytest.raises(SamplerPlanError, match=code):
        build_distributed_sampler_plan(1, **kwargs)
