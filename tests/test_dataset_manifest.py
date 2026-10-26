from dataclasses import replace

import pytest

from structxai.dataset_manifest import (
    MAX_EXAMPLE_BYTES,
    DatasetExample,
    DatasetManifestError,
    build_dataset_manifest,
    validate_dataset_manifest,
    verify_dataset_manifest,
)


def examples() -> list[DatasetExample]:
    return [
        DatasetExample("prompt-1", b'{"prompt":"alpha","target":"A"}'),
        DatasetExample("prompt-2", b'{"prompt":"beta","target":"B"}'),
        DatasetExample("prompt-3", b'{"prompt":"gamma","target":"C"}'),
    ]


def test_manifest_is_deterministic_ordered_and_content_minimizing() -> None:
    values = examples()

    first = build_dataset_manifest(values)
    second = build_dataset_manifest(values)

    assert first == second
    assert first.dataset_size == 3
    assert [record.position for record in first.records] == [0, 1, 2]
    assert first.total_payload_bytes == sum(len(item.payload) for item in values)
    assert all(len(record.example_id_sha256) == 64 for record in first.records)
    assert all(len(record.payload_sha256) == 64 for record in first.records)
    assert "prompt-1" not in repr(first.as_dict())
    assert "alpha" not in repr(first.as_dict())
    assert validate_dataset_manifest(first)


@pytest.mark.parametrize("change", ["identity", "payload", "order"])
def test_manifest_detects_exact_dataset_drift(change: str) -> None:
    values = examples()
    baseline = build_dataset_manifest(values)

    if change == "identity":
        values[0] = DatasetExample("renamed", values[0].payload)
    elif change == "payload":
        values[0] = DatasetExample(values[0].example_id, b"changed")
    else:
        values[0], values[1] = values[1], values[0]

    assert not verify_dataset_manifest(baseline, values)
    assert build_dataset_manifest(values).evidence_sha256 != baseline.evidence_sha256


@pytest.mark.parametrize(
    ("values", "code"),
    [
        ([], "DATASET_SIZE_INVALID"),
        ([DatasetExample("", b"payload")], "DATASET_EXAMPLE_ID_INVALID"),
        ([DatasetExample("id", b"")], "DATASET_PAYLOAD_SIZE_INVALID"),
        (
            [DatasetExample("id", b"a"), DatasetExample("id", b"b")],
            "DATASET_EXAMPLE_ID_DUPLICATE",
        ),
        ([DatasetExample("id", bytearray(b"payload"))], "DATASET_PAYLOAD_INVALID"),
    ],
)
def test_invalid_dataset_examples_fail_closed(values, code: str) -> None:
    with pytest.raises(DatasetManifestError, match=code):
        build_dataset_manifest(values)


def test_oversized_payload_fails_before_manifest_creation() -> None:
    with pytest.raises(DatasetManifestError, match="DATASET_PAYLOAD_SIZE_INVALID"):
        build_dataset_manifest([DatasetExample("id", b"x" * (MAX_EXAMPLE_BYTES + 1))])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda manifest: replace(manifest, evidence_sha256="0" * 64),
        lambda manifest: replace(
            manifest,
            records=(replace(manifest.records[0], position=1), *manifest.records[1:]),
        ),
        lambda manifest: replace(manifest, total_payload_bytes=manifest.total_payload_bytes + 1),
    ],
)
def test_structurally_forged_manifest_is_rejected(mutate) -> None:
    manifest = build_dataset_manifest(examples())

    assert not validate_dataset_manifest(mutate(manifest))
