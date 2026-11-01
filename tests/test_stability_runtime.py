from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from structxai.artifact_manifest import build_artifact_manifest
from structxai.dataset_manifest import DatasetExample, build_dataset_manifest
from structxai.stability_runtime import (
    LockedStabilityInputs,
    StabilityRuntimeError,
    run_manifest_locked_stability_cohort,
)
from tests.test_stability_manifest import _manifest, _reports


def _write_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    model = tmp_path / "model"
    tokenizer = tmp_path / "tokenizer"
    model.mkdir(parents=True)
    tokenizer.mkdir(parents=True)
    (model / "config.json").write_text('{"architecture":"tiny"}\n', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"reviewed-model-weights")
    (tokenizer / "tokenizer.json").write_text('{"version":"1.0"}\n', encoding="utf-8")
    return model, tokenizer


def _examples() -> list[DatasetExample]:
    return [DatasetExample(f"public-task-{index:02d}", f"record-{index}".encode()) for index in range(8)]


def _fixture(tmp_path: Path):
    reports = _reports()
    model, tokenizer = _write_artifacts(tmp_path)
    examples = _examples()
    protocol = b'{"selection":"first-eight-reviewed-tasks","version":1}\n'
    manifest = _manifest(
        reports,
        dataset_records_sha256=build_dataset_manifest(examples).evidence_sha256,
        selection_protocol_sha256=hashlib.sha256(protocol).hexdigest(),
        model_weights_sha256=build_artifact_manifest(model).evidence_sha256,
        tokenizer_sha256=build_artifact_manifest(tokenizer).evidence_sha256,
    )
    return reports, manifest, model, tokenizer, examples, protocol


def _run(tmp_path: Path, runner=None, **overrides):
    reports, manifest, model, tokenizer, examples, protocol = _fixture(tmp_path)
    arguments = {
        "expected_manifest_sha256": manifest["manifest_sha256"],
        "model_path": model,
        "tokenizer_path": tokenizer,
        "dataset_examples": examples,
        "selection_protocol": protocol,
        "runner": runner or (lambda inputs: reports),
    }
    arguments.update(overrides)
    return (
        manifest,
        reports,
        run_manifest_locked_stability_cohort(
            manifest,
            **arguments,
        ),
    )


def test_runtime_gate_verifies_concrete_inputs_before_and_after_callback(
    tmp_path: Path,
) -> None:
    observed: list[LockedStabilityInputs] = []

    def runner(inputs: LockedStabilityInputs):
        observed.append(inputs)
        return _reports()

    manifest, reports, (returned, evidence) = _run(tmp_path, runner=runner)

    assert returned == reports
    assert len(observed) == 1
    assert observed[0].model_path.is_absolute()
    assert observed[0].tokenizer_path.is_absolute()
    assert observed[0].dataset_examples == tuple(_examples())
    assert evidence.postflight_verified is True
    assert evidence.manifest_sha256 == manifest["manifest_sha256"]
    assert evidence.task_count == 8
    assert (
        evidence.evidence_sha256
        == hashlib.sha256(
            json.dumps(
                evidence.body(),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()
    )
    assert "record-0" not in str(evidence.to_audit_dict())
    assert "reviewed-model-weights" not in str(evidence.to_audit_dict())


def test_published_manifest_digest_is_checked_before_runner(tmp_path: Path) -> None:
    called = False

    def runner(inputs):
        nonlocal called
        called = True
        return _reports()

    with pytest.raises(StabilityRuntimeError, match="published digest"):
        _run(tmp_path, runner=runner, expected_manifest_sha256="0" * 64)

    assert called is False


@pytest.mark.parametrize(
    "input_name",
    ["model_path", "tokenizer_path", "dataset_examples", "selection_protocol"],
)
def test_input_drift_is_rejected_before_runner(
    tmp_path: Path,
    input_name: str,
) -> None:
    reports, manifest, model, tokenizer, examples, protocol = _fixture(tmp_path)
    changed_model, changed_tokenizer = _write_artifacts(tmp_path / "changed")
    (changed_model / "model.safetensors").write_bytes(b"different-model")
    (changed_tokenizer / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    changed = {
        "model_path": changed_model,
        "tokenizer_path": changed_tokenizer,
        "dataset_examples": [*examples[:-1], DatasetExample("other", b"other")],
        "selection_protocol": protocol + b"changed",
    }
    called = False

    def runner(inputs):
        nonlocal called
        called = True
        return reports

    arguments = {
        "expected_manifest_sha256": manifest["manifest_sha256"],
        "model_path": model,
        "tokenizer_path": tokenizer,
        "dataset_examples": examples,
        "selection_protocol": protocol,
        "runner": runner,
        input_name: changed[input_name],
    }
    with pytest.raises(StabilityRuntimeError, match="do not match"):
        run_manifest_locked_stability_cohort(manifest, **arguments)

    assert called is False


@pytest.mark.parametrize("target", ["model", "tokenizer"])
def test_artifact_mutation_during_callback_fails_postflight(
    tmp_path: Path,
    target: str,
) -> None:
    reports, manifest, model, tokenizer, examples, protocol = _fixture(tmp_path)

    def runner(inputs: LockedStabilityInputs):
        path = inputs.model_path if target == "model" else inputs.tokenizer_path
        (path / "runtime-mutation.bin").write_bytes(b"mutation")
        return reports

    with pytest.raises(StabilityRuntimeError, match="changed during execution"):
        run_manifest_locked_stability_cohort(
            manifest,
            expected_manifest_sha256=manifest["manifest_sha256"],
            model_path=model,
            tokenizer_path=tokenizer,
            dataset_examples=examples,
            selection_protocol=protocol,
            runner=runner,
        )


def test_symlinked_artifact_root_is_rejected_before_runner(tmp_path: Path) -> None:
    reports, manifest, model, tokenizer, examples, protocol = _fixture(tmp_path)
    link = tmp_path / "model-link"
    link.symlink_to(model, target_is_directory=True)
    called = False

    def runner(inputs):
        nonlocal called
        called = True
        return reports

    with pytest.raises(StabilityRuntimeError, match="model path is invalid"):
        run_manifest_locked_stability_cohort(
            manifest,
            expected_manifest_sha256=manifest["manifest_sha256"],
            model_path=link,
            tokenizer_path=tokenizer,
            dataset_examples=examples,
            selection_protocol=protocol,
            runner=runner,
        )
    assert called is False


@pytest.mark.parametrize("failure", ["missing", "substituted", "malformed"])
def test_callback_reports_must_match_every_committed_task(
    tmp_path: Path,
    failure: str,
) -> None:
    reports, manifest, model, tokenizer, examples, protocol = _fixture(tmp_path)
    changed = dict(reports)
    if failure == "missing":
        changed.pop("public-task-07")
    elif failure == "substituted":
        changed["public-task-07"] = {
            **changed["public-task-07"],
            "experiment_sha256": "0" * 64,
        }
    else:
        changed["public-task-07"] = []

    with pytest.raises(StabilityRuntimeError, match="reports|commitment"):
        run_manifest_locked_stability_cohort(
            manifest,
            expected_manifest_sha256=manifest["manifest_sha256"],
            model_path=model,
            tokenizer_path=tokenizer,
            dataset_examples=examples,
            selection_protocol=protocol,
            runner=lambda inputs: changed,
        )


def test_callback_failure_is_sanitized(tmp_path: Path) -> None:
    def runner(inputs):
        raise RuntimeError("private prompt and model path diagnostic")

    with pytest.raises(StabilityRuntimeError, match="cohort runner failed") as error:
        _run(tmp_path, runner=runner)

    assert "private prompt" not in str(error.value)


def test_callback_reports_are_returned_as_detached_canonical_snapshots(
    tmp_path: Path,
) -> None:
    reports, manifest, model, tokenizer, examples, protocol = _fixture(tmp_path)

    returned, _ = run_manifest_locked_stability_cohort(
        manifest,
        expected_manifest_sha256=manifest["manifest_sha256"],
        model_path=model,
        tokenizer_path=tokenizer,
        dataset_examples=examples,
        selection_protocol=protocol,
        runner=lambda inputs: reports,
    )
    reports["public-task-00"]["accepted"] = False

    assert returned["public-task-00"]["accepted"] is True


@pytest.mark.parametrize("selection_protocol", [b"", b"x" * (16 * 1024**2 + 1)])
def test_selection_protocol_payload_is_bounded(
    tmp_path: Path,
    selection_protocol: bytes,
) -> None:
    with pytest.raises(StabilityRuntimeError, match="payload size"):
        _run(tmp_path, selection_protocol=selection_protocol)
