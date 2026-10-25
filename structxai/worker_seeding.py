from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch

WORKER_SEED_MANIFEST_SCHEMA = "struct-xai-worker-seed-manifest/v1"
MAX_SEED = 2**63 - 1
MAX_WORLD_SIZE = 1_024
MAX_WORKERS_PER_RANK = 256
MAX_TOTAL_WORKERS = 4_096


class WorkerSeedError(ValueError):
    """Stable failure for an invalid or unsafe worker-seed plan."""


@dataclass(frozen=True, slots=True)
class WorkerSeedCommitment:
    rank: int
    worker_id: int
    seed_sha256: str


@dataclass(frozen=True, slots=True)
class WorkerSeedManifest:
    schema_version: str
    world_size: int
    workers_per_rank: int
    epoch: int
    commitments: tuple[WorkerSeedCommitment, ...]
    evidence_sha256: str

    def body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "world_size": self.world_size,
            "workers_per_rank": self.workers_per_rank,
            "epoch": self.epoch,
            "commitments": [asdict(item) for item in self.commitments],
        }

    def as_dict(self) -> dict[str, object]:
        return {**self.body(), "evidence_sha256": self.evidence_sha256}


@dataclass(frozen=True, slots=True)
class WorkerSeedEntry:
    worker_id: int
    seed: int


@dataclass(frozen=True, slots=True)
class WorkerSeedInitializer:
    """Picklable ``DataLoader.worker_init_fn`` for one distributed rank."""

    entries: tuple[WorkerSeedEntry, ...]

    def __call__(self, worker_id: int) -> None:
        if type(worker_id) is not int:
            raise WorkerSeedError("WORKER_ID_INVALID")
        seeds = {entry.worker_id: entry.seed for entry in self.entries}
        try:
            seed = seeds[worker_id]
        except KeyError as exc:
            raise WorkerSeedError("WORKER_ID_OUT_OF_RANGE") from exc
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)


@dataclass(frozen=True, slots=True)
class WorkerSeedPlan:
    rank: int
    manifest: WorkerSeedManifest
    initializer: WorkerSeedInitializer


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


def _validated_int(
    value: object,
    *,
    lower: int,
    upper: int,
    code: str,
) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise WorkerSeedError(code)
    return value


def _derive_seed(root_seed: int, epoch: int, rank: int, worker_id: int) -> int:
    material = {
        "domain": "struct-xai-worker-seed/v1",
        "root_seed": root_seed,
        "epoch": epoch,
        "rank": rank,
        "worker_id": worker_id,
    }
    derived = int.from_bytes(hashlib.sha256(_canonical_json(material)).digest()[:8], "big")
    return derived & MAX_SEED


def build_worker_seed_plan(
    root_seed: int,
    *,
    world_size: int,
    rank: int,
    workers_per_rank: int,
    epoch: int = 0,
) -> WorkerSeedPlan:
    """Derive a bounded global seed manifest and one rank's worker initializer."""

    active_seed = _validated_int(
        root_seed,
        lower=0,
        upper=MAX_SEED,
        code="ROOT_SEED_INVALID",
    )
    active_world_size = _validated_int(
        world_size,
        lower=1,
        upper=MAX_WORLD_SIZE,
        code="WORLD_SIZE_INVALID",
    )
    active_rank = _validated_int(
        rank,
        lower=0,
        upper=active_world_size - 1,
        code="RANK_INVALID",
    )
    active_workers = _validated_int(
        workers_per_rank,
        lower=1,
        upper=MAX_WORKERS_PER_RANK,
        code="WORKER_COUNT_INVALID",
    )
    active_epoch = _validated_int(
        epoch,
        lower=0,
        upper=MAX_SEED,
        code="EPOCH_INVALID",
    )
    total_workers = active_world_size * active_workers
    if total_workers > MAX_TOTAL_WORKERS:
        raise WorkerSeedError("WORKER_TOPOLOGY_LIMIT_EXCEEDED")

    commitments: list[WorkerSeedCommitment] = []
    local_entries: list[WorkerSeedEntry] = []
    seen_seeds: set[int] = set()
    for worker_rank in range(active_world_size):
        for worker_id in range(active_workers):
            seed = _derive_seed(active_seed, active_epoch, worker_rank, worker_id)
            if seed in seen_seeds:
                raise WorkerSeedError("WORKER_SEED_COLLISION")
            seen_seeds.add(seed)
            commitments.append(
                WorkerSeedCommitment(
                    rank=worker_rank,
                    worker_id=worker_id,
                    seed_sha256=_digest(
                        {
                            "domain": "struct-xai-worker-seed-commitment/v1",
                            "seed": seed,
                        }
                    ),
                )
            )
            if worker_rank == active_rank:
                local_entries.append(WorkerSeedEntry(worker_id=worker_id, seed=seed))

    body = {
        "schema_version": WORKER_SEED_MANIFEST_SCHEMA,
        "world_size": active_world_size,
        "workers_per_rank": active_workers,
        "epoch": active_epoch,
        "commitments": [asdict(item) for item in commitments],
    }
    manifest = WorkerSeedManifest(
        schema_version=WORKER_SEED_MANIFEST_SCHEMA,
        world_size=active_world_size,
        workers_per_rank=active_workers,
        epoch=active_epoch,
        commitments=tuple(commitments),
        evidence_sha256=_digest(body),
    )
    return WorkerSeedPlan(
        rank=active_rank,
        manifest=manifest,
        initializer=WorkerSeedInitializer(tuple(local_entries)),
    )


def verify_worker_seed_plan(plan: WorkerSeedPlan, root_seed: int) -> bool:
    """Rebuild a plan so forged topology or initializer state fails closed."""

    if not isinstance(plan, WorkerSeedPlan) or not isinstance(plan.manifest, WorkerSeedManifest):
        return False
    try:
        rebuilt = build_worker_seed_plan(
            root_seed,
            world_size=plan.manifest.world_size,
            rank=plan.rank,
            workers_per_rank=plan.manifest.workers_per_rank,
            epoch=plan.manifest.epoch,
        )
    except WorkerSeedError:
        return False
    return plan == rebuilt
