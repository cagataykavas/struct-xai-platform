from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest

from structxai.tokenizer_migration import (
    MalformedArtifact,
    TokenizerMigrationPolicy,
    audit_tokenizer_migration,
    load_artifact,
)

NOW = datetime(2026, 9, 29, 1, 0, tzinfo=UTC)


def _token(start: int, end: int, attribution: float) -> dict:
    return {"start_byte": start, "end_byte": end, "attribution": attribution}


def _case(case_id: str = "case-1") -> dict:
    return {
        "case_id": case_id,
        "input_digest": "d" * 64,
        "input_byte_length": 4,
        "reference": {
            "tokenizer_digest": "a" * 64,
            "output_margin": 1.25,
            "tokens": [_token(0, 2, 2.0), _token(2, 4, 2.0)],
        },
        "candidate": {
            "tokenizer_digest": "b" * 64,
            "output_margin": 1.25,
            "tokens": [_token(0, 4, 4.0)],
        },
    }


def artifact() -> dict:
    return {
        "schema_version": "struct-xai/tokenizer-migration/v1",
        "created_at": "2026-09-29T00:59:50Z",
        "benchmark_digest": "1" * 64,
        "model_weights_digest": "2" * 64,
        "target_digest": "3" * 64,
        "cases": [_case()],
    }


def audit(value: dict, policy: TokenizerMigrationPolicy | None = None) -> dict:
    return audit_tokenizer_migration(value, policy, now=NOW)


def test_accepts_equivalent_attribution_after_projection() -> None:
    report = audit(artifact())

    assert report["accepted"] is True
    assert report["reason_codes"] == []
    assert report["summary"] == {
        "case_count": 1,
        "failed_case_count": 0,
        "failed_case_fraction": 0.0,
    }
    metric = report["case_metrics"][0]
    assert metric["reference_token_count"] == 2
    assert metric["candidate_token_count"] == 1
    assert metric["signed_cosine"] == pytest.approx(1.0)
    assert metric["relative_l1"] == pytest.approx(0.0)
    assert metric["attribution_sum_drift"] == pytest.approx(0.0)


def test_report_is_deterministic() -> None:
    assert audit(artifact()) == audit(deepcopy(artifact()))


def test_report_does_not_expose_case_or_input_identity() -> None:
    value = artifact()
    value["cases"][0]["candidate"]["tokens"] = [_token(0, 4, -4.0)]

    rendered = json.dumps(audit(value), sort_keys=True)

    assert "case-1" not in rendered
    assert "d" * 64 not in rendered
    assert "SIGNED_COSINE_BELOW_MINIMUM" in rendered


def test_rejects_direction_reversal() -> None:
    value = artifact()
    value["cases"][0]["candidate"]["tokens"] = [_token(0, 4, -4.0)]

    report = audit(value)

    assert report["accepted"] is False
    assert set(report["reason_codes"]) >= {
        "SIGNED_COSINE_BELOW_MINIMUM",
        "RELATIVE_L1_ABOVE_MAXIMUM",
        "FAILED_CASE_FRACTION_EXCEEDED",
    }


def test_rejects_top_k_salience_migration() -> None:
    value = artifact()
    value["cases"][0]["reference"]["tokens"] = [
        _token(0, 2, 4.0),
        _token(2, 4, 0.0),
    ]
    value["cases"][0]["candidate"]["tokens"] = [
        _token(0, 2, 0.0),
        _token(2, 4, 4.0),
    ]

    report = audit(value)

    assert "TOP_K_OVERLAP_BELOW_MINIMUM" in report["reason_codes"]


def test_rejects_output_margin_drift() -> None:
    value = artifact()
    value["cases"][0]["candidate"]["output_margin"] = 1.3

    assert "OUTPUT_MARGIN_DRIFT" in audit(value)["reason_codes"]


def test_rejects_attribution_sum_drift() -> None:
    value = artifact()
    value["cases"][0]["candidate"]["tokens"] = [_token(0, 4, 5.0)]

    assert "ATTRIBUTION_SUM_DRIFT" in audit(value)["reason_codes"]


def test_rejects_degenerate_attribution() -> None:
    value = artifact()
    value["cases"][0]["reference"]["tokens"] = [_token(0, 4, 0.0)]

    report = audit(value)

    assert "DEGENERATE_ATTRIBUTION" in report["reason_codes"]
    assert report["case_metrics"][0]["relative_l1"] is None


def test_failed_case_fraction_can_be_governed() -> None:
    value = artifact()
    failing = _case("case-2")
    failing["candidate"]["output_margin"] = 2.0
    value["cases"].append(failing)

    report = audit(value, TokenizerMigrationPolicy(max_failed_case_fraction=0.5))

    assert report["accepted"] is True
    assert report["summary"]["failed_case_fraction"] == pytest.approx(0.5)
    assert "OUTPUT_MARGIN_DRIFT" in report["reason_codes"]
    assert "FAILED_CASE_FRACTION_EXCEEDED" not in report["reason_codes"]


def test_rejects_stale_artifact() -> None:
    value = artifact()
    value["created_at"] = "2026-09-27T00:00:00Z"

    assert "STALE_ARTIFACT" in audit(value)["reason_codes"]


def test_rejects_artifact_from_future() -> None:
    value = artifact()
    value["created_at"] = "2026-09-29T01:01:01Z"

    assert "ARTIFACT_FROM_FUTURE" in audit(value)["reason_codes"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(schema_version="wrong"), "schema_version"),
        (lambda value: value.update(extra=True), "fields mismatch"),
        (lambda value: value.update(cases=[]), "cannot be empty"),
        (
            lambda value: value.update(benchmark_digest="not-a-digest"),
            "SHA-256",
        ),
        (
            lambda value: value["cases"][0].update(case_id=" invalid"),
            "case_id",
        ),
        (
            lambda value: value["cases"][0].update(input_byte_length=True),
            "integer",
        ),
        (
            lambda value: value["cases"][0]["candidate"].update(output_margin="1.0"),
            "numeric",
        ),
        (
            lambda value: value["cases"][0]["candidate"].update(tokenizer_digest="a" * 64),
            "must differ",
        ),
    ],
)
def test_malformed_shapes_fail_closed(mutate, message: str) -> None:
    value = artifact()
    mutate(value)

    with pytest.raises(MalformedArtifact, match=message):
        audit(value)


@pytest.mark.parametrize(
    "tokens",
    [
        [],
        [_token(1, 4, 4.0)],
        [_token(0, 3, 3.0)],
        [_token(0, 3, 3.0), _token(2, 4, 1.0)],
        [_token(0, 0, 1.0), _token(0, 4, 3.0)],
        [_token(0, 5, 4.0)],
    ],
)
def test_invalid_span_coverage_fails_closed(tokens: list[dict]) -> None:
    value = artifact()
    value["cases"][0]["candidate"]["tokens"] = tokens

    with pytest.raises(MalformedArtifact, match="empty|span|cover|integer"):
        audit(value)


def test_duplicate_case_id_fails_closed() -> None:
    value = artifact()
    value["cases"].append(deepcopy(value["cases"][0]))

    with pytest.raises(MalformedArtifact, match="duplicate case_id"):
        audit(value)


def test_case_and_projection_budgets_fail_closed() -> None:
    value = artifact()
    value["cases"].append(_case("case-2"))
    with pytest.raises(MalformedArtifact, match="item budget"):
        audit(value, TokenizerMigrationPolicy(max_cases=1))

    value = artifact()
    value["cases"][0]["input_byte_length"] = 5
    value["cases"][0]["reference"]["tokens"] = [_token(0, 5, 4.0)]
    value["cases"][0]["candidate"]["tokens"] = [_token(0, 5, 4.0)]
    with pytest.raises(MalformedArtifact, match="projection budget"):
        audit(value, TokenizerMigrationPolicy(max_input_bytes=4))


def test_policy_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="within"):
        audit(artifact(), TokenizerMigrationPolicy(min_signed_cosine=1.1))
    with pytest.raises(ValueError, match="positive integer"):
        audit(artifact(), TokenizerMigrationPolicy(max_cases=0))


def test_loader_rejects_duplicate_json_key(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text('{"cases":[],"cases":[]}', encoding="utf-8")

    with pytest.raises(MalformedArtifact, match="duplicate JSON key"):
        load_artifact(path, TokenizerMigrationPolicy())


def test_loader_rejects_non_finite_number(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text('{"value":NaN}', encoding="utf-8")

    with pytest.raises(MalformedArtifact, match="non-finite"):
        load_artifact(path, TokenizerMigrationPolicy())


def test_loader_rejects_byte_budget(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text(" " * 101, encoding="utf-8")

    with pytest.raises(MalformedArtifact, match="byte budget"):
        load_artifact(path, TokenizerMigrationPolicy(max_artifact_bytes=100))


def test_findings_are_bounded() -> None:
    value = artifact()
    value["cases"][0]["candidate"]["tokens"] = [_token(0, 4, -5.0)]
    value["cases"][0]["candidate"]["output_margin"] = 2.0

    report = audit(value, TokenizerMigrationPolicy(max_reported_findings=2))

    assert report["finding_count"] > 2
    assert len(report["reported_findings"]) == 2
    assert report["truncated_findings"] == report["finding_count"] - 2


def test_cli_exit_codes_and_atomic_output(tmp_path: Path) -> None:
    input_path = tmp_path / "artifact.json"
    output_path = tmp_path / "report.json"
    input_path.write_text(json.dumps(artifact()), encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "structxai.tokenizer_migration",
        str(input_path),
        "--output",
        str(output_path),
        "--max-artifact-age-seconds",
        "31536000",
    ]

    accepted = subprocess.run(command, check=False, capture_output=True)
    assert accepted.returncode == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["accepted"] is True
    assert not list(tmp_path.glob(".report.json.*"))

    rejected_value = artifact()
    rejected_value["cases"][0]["candidate"]["output_margin"] = 2.0
    input_path.write_text(json.dumps(rejected_value), encoding="utf-8")
    rejected = subprocess.run(command, check=False, capture_output=True)
    assert rejected.returncode == 2

    input_path.write_text("{", encoding="utf-8")
    malformed = subprocess.run(command, check=False, capture_output=True)
    assert malformed.returncode == 3
    assert json.loads(malformed.stderr)["error"] == "malformed_artifact"
