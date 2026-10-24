import os
from pathlib import Path

import pytest

from structxai.artifact_manifest import (
    ArtifactManifestError,
    ArtifactManifestPolicy,
    build_artifact_manifest,
    verify_artifact_manifest,
)


def write_artifact(root: Path) -> None:
    (root / "weights").mkdir(parents=True)
    (root / "config.json").write_text('{"model":"tiny"}\n', encoding="utf-8")
    (root / "weights" / "model.safetensors").write_bytes(b"model-weights")


def test_manifest_is_deterministic_relative_and_content_minimizing(tmp_path: Path) -> None:
    artifact = tmp_path / "model"
    write_artifact(artifact)

    first = build_artifact_manifest(artifact)
    second = build_artifact_manifest(artifact)

    assert first == second
    assert [item.relative_path for item in first.files] == [
        "config.json",
        "weights/model.safetensors",
    ]
    assert all(not Path(item.relative_path).is_absolute() for item in first.files)
    assert b"model-weights".decode() not in repr(first)
    assert verify_artifact_manifest(first, artifact) is True


@pytest.mark.parametrize("change", ["content", "name", "addition"])
def test_manifest_detects_artifact_drift(tmp_path: Path, change: str) -> None:
    artifact = tmp_path / "model"
    write_artifact(artifact)
    baseline = build_artifact_manifest(artifact)

    if change == "content":
        (artifact / "config.json").write_text('{"model":"changed"}\n', encoding="utf-8")
    elif change == "name":
        (artifact / "config.json").rename(artifact / "configuration.json")
    else:
        (artifact / "added.txt").write_text("new", encoding="utf-8")

    assert verify_artifact_manifest(baseline, artifact) is False
    assert build_artifact_manifest(artifact).evidence_sha256 != baseline.evidence_sha256


@pytest.mark.parametrize(
    ("policy", "code"),
    [
        (ArtifactManifestPolicy(max_files=1), "ARTIFACT_FILE_LIMIT_EXCEEDED"),
        (
            ArtifactManifestPolicy(max_single_file_bytes=5),
            "ARTIFACT_SINGLE_FILE_LIMIT_EXCEEDED",
        ),
        (ArtifactManifestPolicy(max_total_bytes=15), "ARTIFACT_TOTAL_BYTES_LIMIT_EXCEEDED"),
    ],
)
def test_resource_limits_fail_closed(tmp_path: Path, policy: ArtifactManifestPolicy, code: str) -> None:
    artifact = tmp_path / "model"
    write_artifact(artifact)

    with pytest.raises(ArtifactManifestError, match=code):
        build_artifact_manifest(artifact, policy)


def test_empty_missing_and_symlinked_artifacts_are_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ArtifactManifestError, match="ARTIFACT_EMPTY"):
        build_artifact_manifest(empty)
    with pytest.raises(ArtifactManifestError, match="ARTIFACT_ROOT_INVALID"):
        build_artifact_manifest(tmp_path / "missing")

    artifact = tmp_path / "model"
    outside = tmp_path / "outside.bin"
    artifact.mkdir()
    outside.write_bytes(b"outside")
    (artifact / "weights.bin").symlink_to(outside)
    with pytest.raises(ArtifactManifestError, match="ARTIFACT_SYMLINK_REJECTED"):
        build_artifact_manifest(artifact)


def test_invalid_root_and_policy_types_fail_with_stable_errors(tmp_path: Path) -> None:
    with pytest.raises(ArtifactManifestError, match="ARTIFACT_ROOT_INVALID"):
        build_artifact_manifest("")
    with pytest.raises(ArtifactManifestError, match="ARTIFACT_ROOT_INVALID"):
        build_artifact_manifest(None)  # type: ignore[arg-type]
    with pytest.raises(ArtifactManifestError, match="ARTIFACT_POLICY_INVALID"):
        build_artifact_manifest(tmp_path, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_files"):
        ArtifactManifestPolicy(max_files=True)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO is POSIX-only")
def test_special_files_are_rejected_without_opening_them(tmp_path: Path) -> None:
    artifact = tmp_path / "model"
    artifact.mkdir()
    os.mkfifo(artifact / "pipe")

    with pytest.raises(ArtifactManifestError, match="ARTIFACT_FILE_INVALID"):
        build_artifact_manifest(artifact)
