from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from structxai.artifact_manifest import (
    DEFAULT_ARTIFACT_MANIFEST_POLICY,
    ArtifactManifest,
    ArtifactManifestError,
    ArtifactManifestPolicy,
    build_artifact_manifest,
)
from structxai.dataset_manifest import (
    DatasetExample,
    DatasetManifestError,
    build_dataset_manifest,
)
from structxai.stability_manifest import (
    StabilityTaskManifestError,
    validate_stability_task_manifest,
)

STABILITY_RUNTIME_EVIDENCE_SCHEMA = "struct-xai-stability-runtime-evidence/v1"
_MAX_SELECTION_PROTOCOL_BYTES = 16 * 1024**2
_DIGEST_FIELDS = (
    "dataset_records_sha256",
    "selection_protocol_sha256",
    "model_weights_sha256",
    "tokenizer_sha256",
)


class StabilityRuntimeError(RuntimeError):
    """Stable failure from the manifest-locked stability execution boundary."""

    code = "STABILITY_RUNTIME_REJECTED"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


@dataclass(frozen=True, slots=True)
class LockedStabilityInputs:
    """Concrete, preflight-verified inputs supplied to the cohort callback."""

    manifest_sha256: str
    model_path: Path
    tokenizer_path: Path
    dataset_examples: tuple[DatasetExample, ...]
    selection_protocol: bytes


@dataclass(frozen=True, slots=True)
class StabilityRuntimeEvidence:
    schema_version: str
    cohort_id: str
    manifest_sha256: str
    task_count: int
    dataset_records_sha256: str
    selection_protocol_sha256: str
    model_weights_sha256: str
    tokenizer_sha256: str
    input_commitment_sha256: str
    report_commitments_sha256: str
    postflight_verified: bool
    evidence_sha256: str

    def body(self) -> dict[str, object]:
        document = asdict(self)
        document.pop("evidence_sha256")
        return document

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise StabilityRuntimeError("runtime evidence is not canonical JSON") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _compare_digest(observed: str, expected: object) -> bool:
    return isinstance(expected, str) and hmac.compare_digest(observed, expected)


def _resolved_directory(path: str | Path, name: str) -> Path:
    if not isinstance(path, (str, Path)) or (isinstance(path, str) and not path):
        raise StabilityRuntimeError(f"{name} path is invalid")
    candidate = Path(path)
    try:
        if candidate.is_symlink() or not candidate.is_dir():
            raise StabilityRuntimeError(f"{name} path is invalid")
        return candidate.resolve(strict=True)
    except StabilityRuntimeError:
        raise
    except (OSError, RuntimeError) as exc:
        raise StabilityRuntimeError(f"{name} path is invalid") from exc


def _artifact(
    path: Path,
    policy: ArtifactManifestPolicy,
    name: str,
) -> ArtifactManifest:
    try:
        return build_artifact_manifest(path, policy)
    except ArtifactManifestError as exc:
        raise StabilityRuntimeError(f"{name} artifact verification failed") from exc


def _validate_report_commitments(
    reports: object,
    manifest: Mapping[str, object],
) -> tuple[dict[str, dict], str]:
    if not isinstance(reports, Mapping):
        raise StabilityRuntimeError("cohort callback returned invalid reports")
    expected_tasks = manifest["tasks"]
    assert isinstance(expected_tasks, list)
    expected = {
        task["task_id"]: task["experiment_sha256"] for task in expected_tasks if isinstance(task, dict)
    }
    if set(reports) != set(expected):
        raise StabilityRuntimeError("cohort reports do not match committed tasks")
    normalized: dict[str, dict] = {}
    commitments: list[dict[str, str]] = []
    for task_id in sorted(expected):
        report = reports[task_id]
        if not isinstance(task_id, str) or not isinstance(report, dict):
            raise StabilityRuntimeError("cohort callback returned invalid reports")
        experiment_sha256 = report.get("experiment_sha256")
        if not isinstance(experiment_sha256, str) or not hmac.compare_digest(
            experiment_sha256,
            expected[task_id],
        ):
            raise StabilityRuntimeError("cohort report does not match task commitment")
        try:
            canonical_report = _canonical_json(report)
            normalized_report = json.loads(canonical_report)
        except (json.JSONDecodeError, StabilityRuntimeError) as exc:
            raise StabilityRuntimeError("cohort callback returned invalid reports") from exc
        if not isinstance(normalized_report, dict):
            raise StabilityRuntimeError("cohort callback returned invalid reports")
        normalized[task_id] = normalized_report
        commitments.append(
            {
                "task_id": task_id,
                "experiment_sha256": experiment_sha256,
                "report_sha256": hashlib.sha256(canonical_report).hexdigest(),
            }
        )
    return normalized, _digest(commitments)


def run_manifest_locked_stability_cohort(
    manifest: dict[str, object],
    *,
    expected_manifest_sha256: str,
    model_path: str | Path,
    tokenizer_path: str | Path,
    dataset_examples: Sequence[DatasetExample],
    selection_protocol: bytes,
    runner: Callable[[LockedStabilityInputs], Mapping[str, dict]],
    artifact_policy: ArtifactManifestPolicy = DEFAULT_ARTIFACT_MANIFEST_POLICY,
) -> tuple[dict[str, dict], StabilityRuntimeEvidence]:
    """Verify concrete inputs before and after one committed stability cohort run."""

    try:
        normalized_manifest = validate_stability_task_manifest(manifest)
    except StabilityTaskManifestError as exc:
        raise StabilityRuntimeError("stability manifest is invalid") from exc
    manifest_sha256 = normalized_manifest["manifest_sha256"]
    if not isinstance(expected_manifest_sha256, str) or not _compare_digest(
        str(manifest_sha256), expected_manifest_sha256
    ):
        raise StabilityRuntimeError("manifest does not match the published digest")
    if not isinstance(artifact_policy, ArtifactManifestPolicy):
        raise StabilityRuntimeError("artifact policy is invalid")
    if (
        not isinstance(selection_protocol, bytes)
        or not 1 <= len(selection_protocol) <= _MAX_SELECTION_PROTOCOL_BYTES
    ):
        raise StabilityRuntimeError("selection protocol payload size is invalid")
    if not callable(runner):
        raise StabilityRuntimeError("cohort runner is invalid")
    if isinstance(dataset_examples, (str, bytes)) or not isinstance(dataset_examples, Sequence):
        raise StabilityRuntimeError("dataset examples are invalid")
    locked_examples = tuple(dataset_examples)
    try:
        dataset_manifest = build_dataset_manifest(locked_examples)
    except DatasetManifestError as exc:
        raise StabilityRuntimeError("dataset verification failed") from exc

    resolved_model_path = _resolved_directory(model_path, "model")
    resolved_tokenizer_path = _resolved_directory(tokenizer_path, "tokenizer")
    model_manifest = _artifact(resolved_model_path, artifact_policy, "model")
    tokenizer_manifest = _artifact(
        resolved_tokenizer_path,
        artifact_policy,
        "tokenizer",
    )
    observed = {
        "dataset_records_sha256": dataset_manifest.evidence_sha256,
        "selection_protocol_sha256": hashlib.sha256(selection_protocol).hexdigest(),
        "model_weights_sha256": model_manifest.evidence_sha256,
        "tokenizer_sha256": tokenizer_manifest.evidence_sha256,
    }
    if any(not _compare_digest(observed[field], normalized_manifest[field]) for field in _DIGEST_FIELDS):
        raise StabilityRuntimeError("runtime inputs do not match the manifest")

    task_count = len(normalized_manifest["tasks"])
    input_commitment = {
        "manifest_sha256": manifest_sha256,
        "task_count": task_count,
        **observed,
    }
    locked_inputs = LockedStabilityInputs(
        manifest_sha256=str(manifest_sha256),
        model_path=resolved_model_path,
        tokenizer_path=resolved_tokenizer_path,
        dataset_examples=locked_examples,
        selection_protocol=selection_protocol,
    )
    try:
        reports = runner(locked_inputs)
    except Exception as exc:
        raise StabilityRuntimeError("cohort runner failed") from exc
    normalized_reports, report_commitments_sha256 = _validate_report_commitments(
        reports,
        normalized_manifest,
    )

    if (
        _artifact(resolved_model_path, artifact_policy, "model") != model_manifest
        or _artifact(resolved_tokenizer_path, artifact_policy, "tokenizer") != tokenizer_manifest
    ):
        raise StabilityRuntimeError("runtime artifacts changed during execution")
    try:
        if build_dataset_manifest(locked_examples) != dataset_manifest:
            raise StabilityRuntimeError("runtime dataset changed during execution")
    except DatasetManifestError as exc:
        raise StabilityRuntimeError("runtime dataset changed during execution") from exc

    document = {
        "schema_version": STABILITY_RUNTIME_EVIDENCE_SCHEMA,
        "cohort_id": normalized_manifest["cohort_id"],
        "manifest_sha256": manifest_sha256,
        "task_count": task_count,
        **observed,
        "input_commitment_sha256": _digest(input_commitment),
        "report_commitments_sha256": report_commitments_sha256,
        "postflight_verified": True,
    }
    return normalized_reports, StabilityRuntimeEvidence(
        **document,
        evidence_sha256=_digest(document),
    )
