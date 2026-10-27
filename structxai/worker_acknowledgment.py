from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

from structxai.worker_seeding import WorkerSeedInitializer, WorkerSeedPlan

WORKER_ACKNOWLEDGMENT_SCHEMA = "struct-xai-worker-acknowledgment/v1"
MAX_ACKNOWLEDGMENT_BYTES = 4_096


class WorkerAcknowledgmentError(ValueError):
    """Stable failure for missing, duplicate, or malformed worker evidence."""


@dataclass(frozen=True, slots=True)
class WorkerAcknowledgment:
    rank: int
    worker_id: int
    epoch: int
    seed_sha256: str


@dataclass(frozen=True, slots=True)
class WorkerAcknowledgmentManifest:
    schema_version: str
    worker_seed_manifest_sha256: str
    rank: int
    epoch: int
    challenge_sha256: str
    acknowledgments: tuple[WorkerAcknowledgment, ...]
    evidence_sha256: str

    def body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "worker_seed_manifest_sha256": self.worker_seed_manifest_sha256,
            "rank": self.rank,
            "epoch": self.epoch,
            "challenge_sha256": self.challenge_sha256,
            "acknowledgments": [asdict(item) for item in self.acknowledgments],
        }

    def as_dict(self) -> dict[str, object]:
        return {**self.body(), "evidence_sha256": self.evidence_sha256}


@dataclass(frozen=True, slots=True)
class AcknowledgingWorkerSeedInitializer:
    """Picklable initializer that records one atomic acknowledgment per worker."""

    delegate: WorkerSeedInitializer
    directory: str
    rank: int
    epoch: int
    challenge: str
    commitments: tuple[tuple[int, str], ...]

    def __call__(self, worker_id: int) -> None:
        self.delegate(worker_id)
        commitments = dict(self.commitments)
        try:
            seed_sha256 = commitments[worker_id]
        except (KeyError, TypeError) as exc:
            raise WorkerAcknowledgmentError("WORKER_ACK_ID_INVALID") from exc
        record = {
            "schema_version": WORKER_ACKNOWLEDGMENT_SCHEMA,
            "rank": self.rank,
            "worker_id": worker_id,
            "epoch": self.epoch,
            "seed_sha256": seed_sha256,
            "challenge": self.challenge,
        }
        payload = _canonical_json(record)
        if len(payload) > MAX_ACKNOWLEDGMENT_BYTES:
            raise WorkerAcknowledgmentError("WORKER_ACK_FILE_TOO_LARGE")
        path = Path(self.directory, _worker_filename(worker_id))
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError as exc:
            raise WorkerAcknowledgmentError("WORKER_ACK_DUPLICATE") from exc
        except OSError as exc:
            raise WorkerAcknowledgmentError("WORKER_ACK_WRITE_FAILED") from exc
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise WorkerAcknowledgmentError("WORKER_ACK_WRITE_FAILED")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class WorkerAcknowledgmentSession:
    """Own a private, short-lived worker acknowledgment boundary for one rank."""

    def __init__(self, plan: WorkerSeedPlan) -> None:
        if not isinstance(plan, WorkerSeedPlan):
            raise WorkerAcknowledgmentError("WORKER_ACK_PLAN_INVALID")
        active_challenge = secrets.token_hex(32)
        local_commitments = tuple(
            (item.worker_id, item.seed_sha256) for item in plan.manifest.commitments if item.rank == plan.rank
        )
        expected_worker_ids = tuple(entry.worker_id for entry in plan.initializer.entries)
        if tuple(worker_id for worker_id, _digest in local_commitments) != expected_worker_ids:
            raise WorkerAcknowledgmentError("WORKER_ACK_PLAN_INVALID")

        self._plan = plan
        self._challenge = active_challenge
        self._directory = tempfile.mkdtemp(prefix="structxai-worker-ack-")
        os.chmod(self._directory, 0o700)
        self._manifest: WorkerAcknowledgmentManifest | None = None
        self._closed = False
        self.initializer = AcknowledgingWorkerSeedInitializer(
            delegate=plan.initializer,
            directory=self._directory,
            rank=plan.rank,
            epoch=plan.manifest.epoch,
            challenge=active_challenge,
            commitments=local_commitments,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    @property
    def manifest(self) -> WorkerAcknowledgmentManifest:
        if self._manifest is None:
            raise WorkerAcknowledgmentError("WORKER_ACK_NOT_FINALIZED")
        return self._manifest

    def matches(self, plan: WorkerSeedPlan) -> bool:
        return plan == self._plan and not self._closed

    def finalize(self) -> WorkerAcknowledgmentManifest:
        if self._closed:
            raise WorkerAcknowledgmentError("WORKER_ACK_SESSION_CLOSED")
        if self._manifest is not None:
            return self._manifest

        expected = {
            _worker_filename(worker_id): WorkerAcknowledgment(
                rank=self._plan.rank,
                worker_id=worker_id,
                epoch=self._plan.manifest.epoch,
                seed_sha256=seed_sha256,
            )
            for worker_id, seed_sha256 in self.initializer.commitments
        }
        try:
            entries = {entry.name: entry for entry in os.scandir(self._directory)}
        except OSError as exc:
            raise WorkerAcknowledgmentError("WORKER_ACK_READ_FAILED") from exc
        if set(entries) - set(expected):
            raise WorkerAcknowledgmentError("WORKER_ACK_UNEXPECTED")
        if set(expected) - set(entries):
            raise WorkerAcknowledgmentError("WORKER_ACK_MISSING")

        acknowledgments: list[WorkerAcknowledgment] = []
        for filename, expected_item in sorted(expected.items(), key=lambda item: item[1].worker_id):
            entry = entries[filename]
            try:
                payload = _read_bounded_regular_file(entry.path)
                record = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise WorkerAcknowledgmentError("WORKER_ACK_FILE_INVALID") from exc
            expected_record = {
                "schema_version": WORKER_ACKNOWLEDGMENT_SCHEMA,
                "rank": expected_item.rank,
                "worker_id": expected_item.worker_id,
                "epoch": expected_item.epoch,
                "seed_sha256": expected_item.seed_sha256,
                "challenge": self._challenge,
            }
            if record != expected_record or payload != _canonical_json(expected_record):
                raise WorkerAcknowledgmentError("WORKER_ACK_CONTENT_INVALID")
            acknowledgments.append(expected_item)

        challenge_sha256 = hashlib.sha256(self._challenge.encode("ascii")).hexdigest()
        body = {
            "schema_version": WORKER_ACKNOWLEDGMENT_SCHEMA,
            "worker_seed_manifest_sha256": self._plan.manifest.evidence_sha256,
            "rank": self._plan.rank,
            "epoch": self._plan.manifest.epoch,
            "challenge_sha256": challenge_sha256,
            "acknowledgments": [asdict(item) for item in acknowledgments],
        }
        self._manifest = WorkerAcknowledgmentManifest(
            schema_version=WORKER_ACKNOWLEDGMENT_SCHEMA,
            worker_seed_manifest_sha256=self._plan.manifest.evidence_sha256,
            rank=self._plan.rank,
            epoch=self._plan.manifest.epoch,
            challenge_sha256=challenge_sha256,
            acknowledgments=tuple(acknowledgments),
            evidence_sha256=_digest(body),
        )
        return self._manifest

    def close(self) -> None:
        if not self._closed:
            try:
                shutil.rmtree(self._directory)
            except OSError as exc:
                raise WorkerAcknowledgmentError("WORKER_ACK_CLEANUP_FAILED") from exc
            finally:
                self._closed = True


def verify_worker_acknowledgment_manifest(
    manifest: WorkerAcknowledgmentManifest,
    plan: WorkerSeedPlan,
) -> bool:
    """Verify receipt-ready evidence without requiring the deleted temporary files."""

    if not isinstance(manifest, WorkerAcknowledgmentManifest) or not isinstance(plan, WorkerSeedPlan):
        return False
    expected = tuple(
        WorkerAcknowledgment(
            rank=item.rank,
            worker_id=item.worker_id,
            epoch=plan.manifest.epoch,
            seed_sha256=item.seed_sha256,
        )
        for item in plan.manifest.commitments
        if item.rank == plan.rank
    )
    if (
        manifest.schema_version != WORKER_ACKNOWLEDGMENT_SCHEMA
        or manifest.worker_seed_manifest_sha256 != plan.manifest.evidence_sha256
        or manifest.rank != plan.rank
        or manifest.epoch != plan.manifest.epoch
        or manifest.acknowledgments != expected
        or len(manifest.challenge_sha256) != 64
        or any(character not in "0123456789abcdef" for character in manifest.challenge_sha256)
    ):
        return False
    return manifest.evidence_sha256 == _digest(manifest.body())


def _worker_filename(worker_id: int) -> str:
    return f"worker-{worker_id:03d}.json"


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


def _read_bounded_regular_file(path: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_ACKNOWLEDGMENT_BYTES:
            raise ValueError("invalid acknowledgment file")
        chunks: list[bytes] = []
        total = 0
        while total <= MAX_ACKNOWLEDGMENT_BYTES:
            chunk = os.read(descriptor, MAX_ACKNOWLEDGMENT_BYTES + 1 - total)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        payload = b"".join(chunks)
        if len(payload) != metadata.st_size:
            raise ValueError("acknowledgment file changed while reading")
        return payload
    finally:
        os.close(descriptor)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result
