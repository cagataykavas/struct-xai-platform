from __future__ import annotations

import json
import math
from datetime import UTC, datetime

import pytest

from structxai.path_convergence import (
    ConvergenceBinding,
    ConvergenceFormatError,
    ConvergencePolicy,
    audit_convergence,
    parse_artifact,
)

NOW = datetime(2026, 9, 28, 13, 0, tzinfo=UTC)
MODEL = "a" * 64
METHOD = "b" * 64


def estimate(steps: int, values: list[float], features=None) -> dict[str, object]:
    return {
        "integration_steps": steps,
        "feature_ids": features or ["f1", "f2", "f3"],
        "attributions": values,
    }


def document(cases=None, **updates: object) -> dict[str, object]:
    values: dict[str, object] = {
        "schema_version": 1,
        "benchmark_id": "ig-benchmark-v1",
        "model_digest": MODEL,
        "method_digest": METHOD,
        "created_at": "2026-09-28T12:59:00Z",
        "cases": cases
        or [
            {
                "case_id": "case-1",
                "output_delta": 1.0,
                "estimates": [
                    estimate(8, [0.45, 0.30, 0.20]),
                    estimate(16, [0.47, 0.31, 0.20]),
                    estimate(32, [0.48, 0.31, 0.205]),
                    estimate(64, [0.481, 0.311, 0.207]),
                ],
            }
        ],
    }
    values.update(updates)
    return values


def artifact(**updates: object):
    return parse_artifact(json.dumps(document(**updates)).encode())


def binding(**updates: object) -> ConvergenceBinding:
    values = {
        "benchmark_id": "ig-benchmark-v1",
        "model_digest": MODEL,
        "method_digest": METHOD,
    }
    values.update(updates)
    return ConvergenceBinding(**values)  # type: ignore[arg-type]


def policy(**updates: object) -> ConvergencePolicy:
    values = {
        "max_final_completeness_error": 0.01,
        "max_final_relative_l1_change": 0.01,
        "min_final_cosine": 0.999,
    }
    values.update(updates)
    return ConvergencePolicy(**values)  # type: ignore[arg-type]


def audit(item=None, expected=None, rules=None, now=NOW):
    return audit_convergence(item or artifact(), expected or binding(), rules or policy(), now=now)


def test_accepts_converged_path_attributions() -> None:
    report = audit()
    assert report.accepted
    assert report.failed_case_count == 0
    assert report.cases[0].final_completeness_error == pytest.approx(0.001)
    assert report.cases[0].final_relative_l1_change < 0.01
    assert report.cases[0].final_cosine > 0.999


def test_detects_incomplete_final_attribution() -> None:
    item = document()
    item["cases"][0]["estimates"][-1]["attributions"] = [0.2, 0.2, 0.2]
    report = audit(parse_artifact(json.dumps(item).encode()))
    assert "completeness_not_converged" in report.cases[0].finding_codes
    assert not report.accepted


def test_detects_vector_nonconvergence_despite_completeness() -> None:
    item = document()
    item["cases"][0]["estimates"][-1]["attributions"] = [0.8, 0.1, 0.1]
    row = audit(parse_artifact(json.dumps(item).encode())).cases[0]
    assert "attribution_l1_not_converged" in row.finding_codes
    assert "attribution_direction_not_converged" in row.finding_codes


def test_detects_residual_that_worsens() -> None:
    item = document()
    estimates = item["cases"][0]["estimates"]
    estimates[0]["attributions"] = [0.5, 0.3, 0.2]
    estimates[-1]["attributions"] = [0.481, 0.311, 0.207]
    row = audit(parse_artifact(json.dumps(item).encode())).cases[0]
    assert "completeness_did_not_improve" in row.finding_codes


def test_requires_prespecified_step_schedule() -> None:
    item = document()
    item["cases"][0]["estimates"][0]["integration_steps"] = 4
    row = audit(parse_artifact(json.dumps(item).encode())).cases[0]
    assert "step_schedule_mismatch" in row.finding_codes


def test_requires_exact_feature_alignment() -> None:
    item = document()
    item["cases"][0]["estimates"][-1]["feature_ids"] = ["f2", "f1", "f3"]
    row = audit(parse_artifact(json.dumps(item).encode())).cases[0]
    assert "feature_alignment_mismatch" in row.finding_codes


def test_rejects_negligible_output_delta() -> None:
    item = document()
    item["cases"][0]["output_delta"] = 1e-9
    row = audit(parse_artifact(json.dumps(item).encode())).cases[0]
    assert "output_delta_too_small" in row.finding_codes


@pytest.mark.parametrize(
    ("expected", "code"),
    [
        (binding(benchmark_id="other"), "benchmark_binding_mismatch"),
        (binding(model_digest="c" * 64), "model_binding_mismatch"),
        (binding(method_digest="d" * 64), "method_binding_mismatch"),
    ],
)
def test_binds_benchmark_model_and_method(expected, code) -> None:
    assert code in audit(expected=expected).finding_codes


def test_rejects_stale_and_future_artifacts() -> None:
    stale = audit(artifact(created_at="2026-09-27T12:59:59Z"))
    future = audit(artifact(created_at="2026-09-28T13:00:31Z"))
    assert stale.finding_codes == ("artifact_stale",)
    assert future.finding_codes == ("artifact_future_dated",)


def test_governed_failed_case_fraction() -> None:
    good = document()["cases"][0]
    bad = json.loads(json.dumps(good))
    bad["case_id"] = "case-2"
    bad["estimates"][-1]["attributions"] = [0.2, 0.2, 0.2]
    strict = audit(artifact(cases=[good, bad]))
    tolerant = audit(artifact(cases=[good, bad]), rules=policy(max_failed_case_fraction=0.5))
    assert strict.finding_codes == ("failed_case_fraction_exceeded",)
    assert tolerant.accepted
    assert tolerant.failed_case_fraction == 0.5


def test_report_is_private_and_deterministic_under_case_reordering() -> None:
    first = document()["cases"][0]
    second = json.loads(json.dumps(first))
    second["case_id"] = "secret-case-2"
    left = audit(artifact(cases=[first, second]))
    right = audit(artifact(cases=[second, first]))
    assert left.artifact_digest == right.artifact_digest
    rendered = json.dumps(left.to_dict(), sort_keys=True)
    assert "case-1" not in rendered
    assert "secret-case-2" not in rendered


def test_policy_digest_changes_with_thresholds() -> None:
    assert audit().policy_digest != audit(rules=policy(min_final_cosine=0.9)).policy_digest


@pytest.mark.parametrize(
    "mutation",
    [
        lambda item: item.update(extra="unknown"),
        lambda item: item.update(schema_version=2),
        lambda item: item.update(model_digest="A" * 64),
        lambda item: item.update(created_at="2026-09-28T13:00:00"),
        lambda item: item.update(cases=[]),
        lambda item: item["cases"].append(item["cases"][0]),
        lambda item: item["cases"][0].update(estimates=item["cases"][0]["estimates"][:1]),
        lambda item: item["cases"][0]["estimates"][0].update(integration_steps=True),
        lambda item: item["cases"][0]["estimates"][0].update(feature_ids=["f1", "f1", "f3"]),
        lambda item: item["cases"][0]["estimates"][0].update(attributions=[0.1]),
    ],
)
def test_malformed_artifacts_fail_closed(mutation) -> None:
    item = document()
    mutation(item)
    with pytest.raises(ConvergenceFormatError):
        parse_artifact(json.dumps(item).encode())


def test_duplicate_json_and_nonfinite_values_fail_closed() -> None:
    with pytest.raises(ConvergenceFormatError, match="duplicate"):
        parse_artifact(b'{"schema_version":1,"schema_version":1}')
    raw = json.dumps(document()).replace('"output_delta": 1.0', '"output_delta": NaN')
    with pytest.raises(ConvergenceFormatError, match="non-finite"):
        parse_artifact(raw.encode())


def test_byte_budget_is_checked_before_parsing() -> None:
    with pytest.raises(ConvergenceFormatError, match="byte size"):
        parse_artifact(b" " * (2 * 1024 * 1024 + 1))


def test_now_must_be_timezone_aware() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        audit(now=datetime.fromisoformat("2026-09-28T13:00:00"))


@pytest.mark.parametrize(
    "updates",
    [
        {"required_steps": (16,)},
        {"required_steps": (16, 8)},
        {"max_final_completeness_error": math.nan},
        {"max_failed_case_fraction": 1.1},
        {"min_final_cosine": -2.0},
        {"min_abs_output_delta": 0.0},
    ],
)
def test_invalid_policy_fails_at_construction(updates) -> None:
    with pytest.raises(ValueError):
        policy(**updates)


def test_zero_vectors_have_defined_convergence() -> None:
    cases = [
        {
            "case_id": "zero",
            "output_delta": 1.0,
            "estimates": [estimate(step, [0.0, 0.0, 0.0]) for step in (8, 16, 32, 64)],
        }
    ]
    row = audit(artifact(cases=cases)).cases[0]
    assert row.final_cosine == 1.0
    assert "completeness_not_converged" in row.finding_codes
