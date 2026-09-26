from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime

import pytest

from structxai.seed_stability_audit import (
    ArtifactError,
    StabilityPolicy,
    audit_artifact,
    load_artifact,
    main,
)

NOW = datetime(2026, 9, 26, 1, 0, tzinfo=UTC)
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def _run(seed: int, values: tuple[float, ...], *, margin: float = 2.0) -> dict:
    return {
        "seed": seed,
        "output_margin": margin,
        "features": [
            {"feature_id": f"token:{index}", "attribution": value} for index, value in enumerate(values)
        ],
    }


def _artifact(*, unstable: bool = False) -> dict:
    stable_values = [
        (0.90, -0.70, 0.40, 0.10),
        (0.92, -0.69, 0.39, 0.11),
        (0.89, -0.72, 0.42, 0.09),
        (0.91, -0.71, 0.41, 0.10),
        (0.90, -0.70, 0.41, 0.09),
    ]
    if unstable:
        stable_values[-1] = (-0.05, 0.10, -0.20, 1.20)
    return {
        "schema": "struct-xai-attribution-seed-stability/v1",
        "generated_at": "2026-09-26T00:59:00Z",
        "benchmark_id": "tr-xai-mini-v2",
        "model_sha256": DIGEST_A,
        "method_id": "integrated-gradients",
        "method_config_sha256": DIGEST_B,
        "cases": [
            {
                "case_id": "case-private-001",
                "runs": [_run(index + 10, values) for index, values in enumerate(stable_values)],
            }
        ],
    }


def _audit(payload: dict, policy: StabilityPolicy | None = None) -> dict:
    return audit_artifact(payload, policy=policy, now=NOW)


def test_stable_runs_are_accepted_without_raw_identifiers():
    report = _audit(_artifact())
    assert report["accepted"] is True
    assert report["summary"] == {
        "case_count": 1,
        "failed_case_count": 0,
        "failed_case_fraction": 0.0,
        "pair_comparisons": 10,
    }
    encoded = json.dumps(report)
    assert "case-private-001" not in encoded
    assert "tr-xai-mini-v2" not in encoded
    assert "integrated-gradients" not in encoded


def test_unstable_seed_is_policy_rejected_with_stable_reason_codes():
    report = _audit(_artifact(unstable=True))
    assert report["accepted"] is False
    reasons = report["cases"][0]["reasons"]
    assert "pairwise_cosine_below_threshold" in reasons
    assert "top_k_overlap_below_threshold" in reasons
    assert "material_sign_agreement_below_threshold" in reasons


def test_margin_drift_is_detected_separately_from_attribution_drift():
    payload = _artifact()
    payload["cases"][0]["runs"][-1]["output_margin"] = 2.01
    report = _audit(payload)
    assert report["cases"][0]["reasons"] == ["output_margin_drift_above_threshold"]


def test_governed_failed_case_fraction_can_admit_one_failure():
    payload = _artifact()
    second_case = deepcopy(payload["cases"][0])
    second_case["case_id"] = "case-private-002"
    second_case["runs"][-1] = _run(99, (-1.0, 0.0, 0.0, 0.0))
    payload["cases"].append(second_case)
    report = _audit(payload, StabilityPolicy(max_failed_case_fraction=0.5))
    assert report["accepted"] is True
    assert report["summary"]["failed_case_count"] == 1


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda item: item.update(schema="wrong"), "schema"),
        (lambda item: item.update(model_sha256="not-a-digest"), "model_sha256"),
        (lambda item: item.update(generated_at="2026-09-26T00:59:00"), "timezone"),
        (lambda item: item["cases"].clear(), "case bounds"),
        (lambda item: item["cases"][0]["runs"].pop(), "run bounds"),
        (
            lambda item: item["cases"][0]["runs"][1].update(seed=item["cases"][0]["runs"][0]["seed"]),
            "duplicate seeds",
        ),
        (
            lambda item: item["cases"][0]["runs"][1]["features"].pop(),
            "exact same feature IDs",
        ),
        (
            lambda item: item["cases"][0]["runs"][0]["features"].append(
                {"feature_id": "token:0", "attribution": 0.1}
            ),
            "duplicate feature_id",
        ),
        (
            lambda item: item["cases"][0]["runs"][0]["features"][0].update(attribution=float("nan")),
            "canonical JSON",
        ),
    ],
)
def test_malformed_evidence_fails_closed(mutation, match):
    payload = _artifact()
    mutation(payload)
    with pytest.raises(ArtifactError, match=match):
        _audit(payload)


def test_zero_attribution_vector_is_rejected():
    payload = _artifact()
    payload["cases"][0]["runs"][0] = _run(10, (0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ArtifactError, match="non-zero vector"):
        _audit(payload)


def test_duplicate_case_identity_is_rejected():
    payload = _artifact()
    payload["cases"].append(deepcopy(payload["cases"][0]))
    with pytest.raises(ArtifactError, match="case_id values must be unique"):
        _audit(payload)


def test_pair_comparison_budget_is_enforced():
    with pytest.raises(ArtifactError, match="max_pair_comparisons"):
        _audit(_artifact(), StabilityPolicy(max_pair_comparisons=9))


def test_top_k_cannot_exceed_feature_count():
    with pytest.raises(ArtifactError, match="fewer features than top_k"):
        _audit(_artifact(), StabilityPolicy(top_k=5))


def test_stale_and_future_evidence_are_rejected():
    stale = _artifact()
    stale["generated_at"] = "2026-09-18T00:00:00Z"
    with pytest.raises(ArtifactError, match="stale"):
        _audit(stale)
    future = _artifact()
    future["generated_at"] = "2026-09-26T01:06:00Z"
    with pytest.raises(ArtifactError, match="future"):
        _audit(future)


def test_report_and_evidence_digest_are_deterministic():
    left = _audit(_artifact())
    right = _audit(deepcopy(_artifact()))
    assert left == right
    assert len(left["evidence_sha256"]) == 64


def test_load_rejects_duplicate_keys_nonfinite_and_oversize(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    with pytest.raises(ArtifactError, match="duplicate JSON key"):
        load_artifact(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ArtifactError, match="non-finite JSON"):
        load_artifact(nonfinite)
    oversize = tmp_path / "oversize.json"
    oversize.write_text("{} ", encoding="utf-8")
    with pytest.raises(ArtifactError, match="byte budget"):
        load_artifact(oversize, max_bytes=2)


def test_cli_accepts_and_writes_atomic_report(tmp_path, monkeypatch, capsys):
    artifact_path = tmp_path / "artifact.json"
    output_path = tmp_path / "nested" / "report.json"
    artifact_path.write_text(json.dumps(_artifact()), encoding="utf-8")
    monkeypatch.setattr(
        "structxai.seed_stability_audit.datetime",
        type("FixedDatetime", (datetime,), {"now": classmethod(lambda cls, tz=None: NOW)}),
    )
    assert main([str(artifact_path), "--output", str(output_path)]) == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["accepted"] is True
    assert json.loads(capsys.readouterr().out)["accepted"] is True
    assert list(output_path.parent.glob(f".{output_path.name}.*")) == []


def test_cli_distinguishes_policy_rejection_and_malformed_input(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "structxai.seed_stability_audit.datetime",
        type("FixedDatetime", (datetime,), {"now": classmethod(lambda cls, tz=None: NOW)}),
    )
    rejected = tmp_path / "rejected.json"
    rejected.write_text(json.dumps(_artifact(unstable=True)), encoding="utf-8")
    assert main([str(rejected)]) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "seed_stability_policy_rejected"

    malformed = tmp_path / "malformed.json"
    malformed.write_text("not-json", encoding="utf-8")
    assert main([str(malformed)]) == 3
    assert json.loads(capsys.readouterr().out)["reason"] == "malformed_artifact"


@pytest.mark.parametrize(
    "policy",
    [
        StabilityPolicy(min_runs_per_case=True),
        StabilityPolicy(min_pairwise_cosine=float("nan")),
        StabilityPolicy(max_failed_case_fraction=1.1),
        StabilityPolicy(zero_epsilon=0.0),
    ],
)
def test_invalid_policy_fails_closed(policy):
    with pytest.raises(ArtifactError):
        _audit(_artifact(), policy)
