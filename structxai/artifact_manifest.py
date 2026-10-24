from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path

ARTIFACT_MANIFEST_SCHEMA = "struct-xai-artifact-manifest/v1"


class ArtifactManifestError(ValueError):
    """Stable failure for an artifact that cannot be hashed safely."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ArtifactManifestPolicy:
    max_files: int = 10_000
    max_single_file_bytes: int = 10 * 1024**3
    max_total_bytes: int = 50 * 1024**3
    read_chunk_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        limits = (
            ("max_files", self.max_files, 1, 1_000_000),
            ("max_single_file_bytes", self.max_single_file_bytes, 1, 1024**4),
            ("max_total_bytes", self.max_total_bytes, 1, 10 * 1024**4),
            ("read_chunk_bytes", self.read_chunk_bytes, 4_096, 16 * 1024**2),
        )
        for name, value, lower, upper in limits:
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"{name} must be between {lower} and {upper}")


DEFAULT_ARTIFACT_MANIFEST_POLICY = ArtifactManifestPolicy()


@dataclass(frozen=True)
class ArtifactFile:
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ArtifactManifest:
    schema_version: str
    files: tuple[ArtifactFile, ...]
    total_bytes: int
    evidence_sha256: str

    def body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "files": [asdict(item) for item in self.files],
            "total_bytes": self.total_bytes,
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


def _hash_file(path: Path, expected_size: int, chunk_bytes: int) -> str:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ArtifactManifestError("ARTIFACT_READ_FAILED") from exc

    digest = hashlib.sha256()
    bytes_read = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise ArtifactManifestError("ARTIFACT_CHANGED_DURING_HASH")
        while True:
            block = os.read(descriptor, chunk_bytes)
            if not block:
                break
            digest.update(block)
            bytes_read += len(block)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ArtifactManifestError("ARTIFACT_READ_FAILED") from exc
    finally:
        os.close(descriptor)

    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or bytes_read != expected_size:
        raise ArtifactManifestError("ARTIFACT_CHANGED_DURING_HASH")
    return digest.hexdigest()


def build_artifact_manifest(
    root: str | Path,
    policy: ArtifactManifestPolicy = DEFAULT_ARTIFACT_MANIFEST_POLICY,
) -> ArtifactManifest:
    """Hash a bounded regular-file tree without following symbolic links."""

    if not isinstance(policy, ArtifactManifestPolicy):
        raise ArtifactManifestError("ARTIFACT_POLICY_INVALID")
    if not isinstance(root, (str, Path)) or (isinstance(root, str) and not root):
        raise ArtifactManifestError("ARTIFACT_ROOT_INVALID")
    root_path = Path(root)
    try:
        if root_path.is_symlink():
            raise ArtifactManifestError("ARTIFACT_ROOT_SYMLINK_REJECTED")
        if not root_path.is_dir():
            raise ArtifactManifestError("ARTIFACT_ROOT_INVALID")
        root_path = root_path.resolve(strict=True)
        entries = sorted(root_path.rglob("*"), key=lambda item: item.relative_to(root_path).as_posix())
    except ArtifactManifestError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise ArtifactManifestError("ARTIFACT_ROOT_INVALID") from exc

    files: list[tuple[str, Path, int]] = []
    total_bytes = 0
    for entry in entries:
        try:
            relative_path = entry.relative_to(root_path).as_posix()
            if entry.is_symlink():
                raise ArtifactManifestError("ARTIFACT_SYMLINK_REJECTED")
            if entry.is_dir():
                continue
            metadata = entry.stat(follow_symlinks=False)
        except ArtifactManifestError:
            raise
        except OSError as exc:
            raise ArtifactManifestError("ARTIFACT_READ_FAILED") from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise ArtifactManifestError("ARTIFACT_FILE_INVALID")
        if len(files) >= policy.max_files:
            raise ArtifactManifestError("ARTIFACT_FILE_LIMIT_EXCEEDED")
        if metadata.st_size > policy.max_single_file_bytes:
            raise ArtifactManifestError("ARTIFACT_SINGLE_FILE_LIMIT_EXCEEDED")
        total_bytes += metadata.st_size
        if total_bytes > policy.max_total_bytes:
            raise ArtifactManifestError("ARTIFACT_TOTAL_BYTES_LIMIT_EXCEEDED")
        files.append((relative_path, entry, metadata.st_size))

    if not files:
        raise ArtifactManifestError("ARTIFACT_EMPTY")

    manifest_files = tuple(
        ArtifactFile(
            relative_path=relative_path,
            size_bytes=size_bytes,
            sha256=_hash_file(path, size_bytes, policy.read_chunk_bytes),
        )
        for relative_path, path, size_bytes in files
    )
    body = {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA,
        "files": [asdict(item) for item in manifest_files],
        "total_bytes": total_bytes,
    }
    return ArtifactManifest(
        schema_version=ARTIFACT_MANIFEST_SCHEMA,
        files=manifest_files,
        total_bytes=total_bytes,
        evidence_sha256=hashlib.sha256(_canonical_json(body)).hexdigest(),
    )


def verify_artifact_manifest(
    manifest: ArtifactManifest,
    root: str | Path,
    policy: ArtifactManifestPolicy = DEFAULT_ARTIFACT_MANIFEST_POLICY,
) -> bool:
    """Rebuild an artifact manifest and compare every committed file."""

    if not isinstance(manifest, ArtifactManifest):
        return False
    try:
        rebuilt = build_artifact_manifest(root, policy)
    except ArtifactManifestError:
        return False
    return manifest == rebuilt
