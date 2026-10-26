from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import torch
from torch.utils.data import Sampler

from structxai.dataset_manifest import DatasetManifest, validate_dataset_manifest

SAMPLER_MANIFEST_SCHEMA = "struct-xai-distributed-sampler/v2"
MAX_SEED = 2**63 - 1
MAX_WORLD_SIZE = 1_024
MAX_DATASET_SIZE = 1_000_000


class SamplerPlanError(ValueError):
    """Stable failure for an invalid or mutated distributed sampler plan."""


@dataclass(frozen=True, slots=True)
class RankOrderCommitment:
    rank: int
    sample_count: int
    order_sha256: str


@dataclass(frozen=True, slots=True)
class SamplerManifest:
    schema_version: str
    world_size: int
    dataset_size: int
    dataset_bound: bool
    dataset_manifest_sha256: str
    epoch: int
    shuffle: bool
    drop_last: bool
    samples_per_rank: int
    total_size: int
    rank_orders: tuple[RankOrderCommitment, ...]
    evidence_sha256: str

    def body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "world_size": self.world_size,
            "dataset_size": self.dataset_size,
            "dataset_bound": self.dataset_bound,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "epoch": self.epoch,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "samples_per_rank": self.samples_per_rank,
            "total_size": self.total_size,
            "rank_orders": [asdict(item) for item in self.rank_orders],
        }

    def as_dict(self) -> dict[str, object]:
        return {**self.body(), "evidence_sha256": self.evidence_sha256}


@dataclass(frozen=True, slots=True)
class DeterministicRankSampler(Sampler[int]):
    """Immutable rank-local sample order compatible with ``DataLoader``."""

    indices: tuple[int, ...]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


@dataclass(frozen=True, slots=True)
class DistributedSamplerPlan:
    rank: int
    manifest: SamplerManifest
    sampler: DeterministicRankSampler
    dataset_manifest: DatasetManifest | None = None


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _validated_int(value: object, *, lower: int, upper: int, code: str) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise SamplerPlanError(code)
    return value


def _permutation_seed(root_seed: int, epoch: int) -> int:
    material = {
        "domain": "struct-xai-distributed-sampler-order/v1",
        "root_seed": root_seed,
        "epoch": epoch,
    }
    return int.from_bytes(hashlib.sha256(_canonical_json(material)).digest()[:8], "big")


def _global_order(
    root_seed: int,
    dataset_size: int,
    epoch: int,
    shuffle: bool,
    total_size: int,
) -> list[int]:
    if shuffle:
        generator = torch.Generator().manual_seed(_permutation_seed(root_seed, epoch))
        indices = torch.randperm(dataset_size, generator=generator).tolist()
    else:
        indices = list(range(dataset_size))
    if total_size > dataset_size:
        repeats, remainder = divmod(total_size - dataset_size, dataset_size)
        indices.extend(indices * repeats)
        indices.extend(indices[:remainder])
    return indices[:total_size]


def build_distributed_sampler_plan(
    root_seed: int,
    *,
    world_size: int,
    rank: int,
    dataset_size: int,
    epoch: int = 0,
    shuffle: bool = True,
    drop_last: bool = False,
    dataset_manifest: DatasetManifest | None = None,
) -> DistributedSamplerPlan:
    """Build a shared order manifest and one immutable rank-local sampler."""

    active_seed = _validated_int(
        root_seed,
        lower=0,
        upper=MAX_SEED,
        code="SAMPLER_ROOT_SEED_INVALID",
    )
    active_world_size = _validated_int(
        world_size,
        lower=1,
        upper=MAX_WORLD_SIZE,
        code="SAMPLER_WORLD_SIZE_INVALID",
    )
    active_rank = _validated_int(
        rank,
        lower=0,
        upper=active_world_size - 1,
        code="SAMPLER_RANK_INVALID",
    )
    active_dataset_size = _validated_int(
        dataset_size,
        lower=1,
        upper=MAX_DATASET_SIZE,
        code="SAMPLER_DATASET_SIZE_INVALID",
    )
    active_epoch = _validated_int(
        epoch,
        lower=0,
        upper=MAX_SEED,
        code="SAMPLER_EPOCH_INVALID",
    )
    if type(shuffle) is not bool:
        raise SamplerPlanError("SAMPLER_SHUFFLE_INVALID")
    if type(drop_last) is not bool:
        raise SamplerPlanError("SAMPLER_DROP_LAST_INVALID")
    if dataset_manifest is None:
        dataset_bound = False
        dataset_manifest_sha256 = _digest(
            {
                "schema_version": "struct-xai-dataset-manifest/none",
                "dataset_size": active_dataset_size,
            }
        )
    else:
        if not validate_dataset_manifest(dataset_manifest):
            raise SamplerPlanError("SAMPLER_DATASET_MANIFEST_INVALID")
        if dataset_manifest.dataset_size != active_dataset_size:
            raise SamplerPlanError("SAMPLER_DATASET_SIZE_MISMATCH")
        dataset_bound = True
        dataset_manifest_sha256 = dataset_manifest.evidence_sha256

    if drop_last:
        samples_per_rank = active_dataset_size // active_world_size
    else:
        samples_per_rank = (active_dataset_size + active_world_size - 1) // active_world_size
    total_size = samples_per_rank * active_world_size
    order = _global_order(
        active_seed,
        active_dataset_size,
        active_epoch,
        shuffle,
        total_size,
    )
    rank_indices = tuple(
        tuple(order[current_rank:total_size:active_world_size]) for current_rank in range(active_world_size)
    )
    commitments = tuple(
        RankOrderCommitment(
            rank=current_rank,
            sample_count=len(indices),
            order_sha256=_digest(
                {
                    "domain": "struct-xai-rank-order-commitment/v1",
                    "rank": current_rank,
                    "indices": indices,
                }
            ),
        )
        for current_rank, indices in enumerate(rank_indices)
    )
    body = {
        "schema_version": SAMPLER_MANIFEST_SCHEMA,
        "world_size": active_world_size,
        "dataset_size": active_dataset_size,
        "dataset_bound": dataset_bound,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "epoch": active_epoch,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "samples_per_rank": samples_per_rank,
        "total_size": total_size,
        "rank_orders": [asdict(item) for item in commitments],
    }
    manifest = SamplerManifest(
        schema_version=SAMPLER_MANIFEST_SCHEMA,
        world_size=active_world_size,
        dataset_size=active_dataset_size,
        dataset_bound=dataset_bound,
        dataset_manifest_sha256=dataset_manifest_sha256,
        epoch=active_epoch,
        shuffle=shuffle,
        drop_last=drop_last,
        samples_per_rank=samples_per_rank,
        total_size=total_size,
        rank_orders=commitments,
        evidence_sha256=_digest(body),
    )
    return DistributedSamplerPlan(
        rank=active_rank,
        manifest=manifest,
        sampler=DeterministicRankSampler(rank_indices[active_rank]),
        dataset_manifest=dataset_manifest,
    )


def verify_distributed_sampler_plan(
    plan: DistributedSamplerPlan,
    root_seed: int,
) -> bool:
    """Rebuild a plan so forged manifests or mutated orders fail closed."""

    if not isinstance(plan, DistributedSamplerPlan) or not isinstance(plan.manifest, SamplerManifest):
        return False
    try:
        rebuilt = build_distributed_sampler_plan(
            root_seed,
            world_size=plan.manifest.world_size,
            rank=plan.rank,
            dataset_size=plan.manifest.dataset_size,
            epoch=plan.manifest.epoch,
            shuffle=plan.manifest.shuffle,
            drop_last=plan.manifest.drop_last,
            dataset_manifest=plan.dataset_manifest,
        )
    except SamplerPlanError:
        return False
    return plan == rebuilt
