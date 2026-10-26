from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass

DATASET_MANIFEST_SCHEMA = "struct-xai-dataset-manifest/v1"
MAX_EXAMPLES = 1_000_000
MAX_EXAMPLE_ID_BYTES = 512
MAX_EXAMPLE_BYTES = 16 * 1024**2
MAX_TOTAL_BYTES = 50 * 1024**3
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class DatasetManifestError(ValueError):
    """Stable failure for dataset contents that cannot be committed safely."""


@dataclass(frozen=True, slots=True)
class DatasetExample:
    """One caller-canonicalized example; payload bytes never enter audit output."""

    example_id: str
    payload: bytes


@dataclass(frozen=True, slots=True)
class DatasetRecordCommitment:
    position: int
    example_id_sha256: str
    payload_sha256: str
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    schema_version: str
    records: tuple[DatasetRecordCommitment, ...]
    total_payload_bytes: int
    evidence_sha256: str

    @property
    def dataset_size(self) -> int:
        return len(self.records)

    def body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "records": [asdict(record) for record in self.records],
            "total_payload_bytes": self.total_payload_bytes,
        }

    def as_dict(self) -> dict[str, object]:
        return {**self.body(), "evidence_sha256": self.evidence_sha256}


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


def build_dataset_manifest(examples: Sequence[DatasetExample]) -> DatasetManifest:
    """Commit exact ordered IDs and payload bytes without retaining either value."""

    if isinstance(examples, (str, bytes)) or not isinstance(examples, Sequence):
        raise DatasetManifestError("DATASET_EXAMPLES_INVALID")
    if not 1 <= len(examples) <= MAX_EXAMPLES:
        raise DatasetManifestError("DATASET_SIZE_INVALID")

    records: list[DatasetRecordCommitment] = []
    seen_ids: set[str] = set()
    total_payload_bytes = 0
    for position, example in enumerate(examples):
        if not isinstance(example, DatasetExample):
            raise DatasetManifestError("DATASET_EXAMPLE_INVALID")
        if not isinstance(example.example_id, str):
            raise DatasetManifestError("DATASET_EXAMPLE_ID_INVALID")
        example_id_bytes = example.example_id.encode("utf-8")
        if not 1 <= len(example_id_bytes) <= MAX_EXAMPLE_ID_BYTES:
            raise DatasetManifestError("DATASET_EXAMPLE_ID_INVALID")
        if example.example_id in seen_ids:
            raise DatasetManifestError("DATASET_EXAMPLE_ID_DUPLICATE")
        if not isinstance(example.payload, bytes):
            raise DatasetManifestError("DATASET_PAYLOAD_INVALID")
        if not 1 <= len(example.payload) <= MAX_EXAMPLE_BYTES:
            raise DatasetManifestError("DATASET_PAYLOAD_SIZE_INVALID")
        total_payload_bytes += len(example.payload)
        if total_payload_bytes > MAX_TOTAL_BYTES:
            raise DatasetManifestError("DATASET_TOTAL_BYTES_EXCEEDED")
        seen_ids.add(example.example_id)
        records.append(
            DatasetRecordCommitment(
                position=position,
                example_id_sha256=hashlib.sha256(example_id_bytes).hexdigest(),
                payload_sha256=hashlib.sha256(example.payload).hexdigest(),
                payload_bytes=len(example.payload),
            )
        )

    body = {
        "schema_version": DATASET_MANIFEST_SCHEMA,
        "records": [asdict(record) for record in records],
        "total_payload_bytes": total_payload_bytes,
    }
    return DatasetManifest(
        schema_version=DATASET_MANIFEST_SCHEMA,
        records=tuple(records),
        total_payload_bytes=total_payload_bytes,
        evidence_sha256=_digest(body),
    )


def validate_dataset_manifest(manifest: DatasetManifest) -> bool:
    """Validate structure and digest without access to raw examples."""

    if (
        not isinstance(manifest, DatasetManifest)
        or manifest.schema_version != DATASET_MANIFEST_SCHEMA
        or not isinstance(manifest.records, tuple)
        or not 1 <= len(manifest.records) <= MAX_EXAMPLES
        or type(manifest.total_payload_bytes) is not int
        or not 1 <= manifest.total_payload_bytes <= MAX_TOTAL_BYTES
        or not isinstance(manifest.evidence_sha256, str)
        or SHA256_RE.fullmatch(manifest.evidence_sha256) is None
    ):
        return False
    seen_ids: set[str] = set()
    total_payload_bytes = 0
    for position, record in enumerate(manifest.records):
        if (
            not isinstance(record, DatasetRecordCommitment)
            or type(record.position) is not int
            or record.position != position
            or not isinstance(record.example_id_sha256, str)
            or SHA256_RE.fullmatch(record.example_id_sha256) is None
            or record.example_id_sha256 in seen_ids
            or not isinstance(record.payload_sha256, str)
            or SHA256_RE.fullmatch(record.payload_sha256) is None
            or type(record.payload_bytes) is not int
            or not 1 <= record.payload_bytes <= MAX_EXAMPLE_BYTES
        ):
            return False
        seen_ids.add(record.example_id_sha256)
        total_payload_bytes += record.payload_bytes
    if total_payload_bytes != manifest.total_payload_bytes:
        return False
    return _digest(manifest.body()) == manifest.evidence_sha256


def verify_dataset_manifest(
    manifest: DatasetManifest,
    examples: Sequence[DatasetExample],
) -> bool:
    """Rebuild a manifest so changed identities, order, or payloads fail closed."""

    if not validate_dataset_manifest(manifest):
        return False
    try:
        rebuilt = build_dataset_manifest(examples)
    except DatasetManifestError:
        return False
    return manifest == rebuilt
