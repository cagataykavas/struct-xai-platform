from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy

import pytest

from structxai.candidate_swap import (
    CandidateSwapInputError,
    CandidateSwapPolicy,
    audit_candidate_swaps,
    load_artifact,
)


def _observation(order: list[str], margin: float, values: dict[str, float]) -> dict:
    return {
        "candidate_order": order,
        "margin": margin,
        "attributions": [{"feature_id": feature_id, "value": value} for feature_id, value in values.items()],
    }


def _case(case_id: str, scale: float = 1.0) -> dict:
    forward = {"prompt:0": 0.7 * scale, "prompt:1": -0.2 * scale, "prompt:2": 0.1 * scale}
    swapped = {key: -value for key, value in reversed(list(forward.items()))}
    return {
        "case_id": case_id,
        "candidate_a": "candidate-a",
        "candidate_b": "candidate-b",
        "forward": _observation(["candidate-a", "candidate-b"], 1.2 * scale, forward),
        "swapped": _observation(["candidate-b", "candidate-a"], -1.2 * scale, swapped),
    }


def _artifact() -> dict:
    return {"schema_version": 1, "cases": [_case("case-c", 0.8), _case("case-a"), _case("case-b", 1.1)]}


def test_accepts_antisymmetric_candidate_swap_evidence() -> None:
    report = audit_candidate_swaps(_artifact())

    assert report["accepted"] is True
    assert report["summary"] == {
        "passed_cases": 3,
        "failed_cases": 0,
        "case_pass_rate": 1.0,
        "finding_counts": {},
    }
    assert [case["case_id"] for case in report["cases"]] == ["case-a", "case-b", "case-c"]
    assert report["cases"][0]["metrics"]["anti_cosine_similarity"] == pytest.approx(1.0)


def test_rejects_margin_that_does_not_change_sign() -> None:
    artifact = _artifact()
    artifact["cases"][0]["swapped"]["margin"] = 1.2

    report = audit_candidate_swaps(artifact)

    assert report["accepted"] is False
    assert "MARGIN_ANTISYMMETRY_FAILED" in report["cases"][2]["findings"]
    assert report["release_findings"] == ["CASE_PASS_RATE_BELOW_POLICY"]


def test_rejects_attributions_that_do_not_change_sign() -> None:
    artifact = _artifact()
    for item in artifact["cases"][0]["swapped"]["attributions"]:
        item["value"] *= -1

    report = audit_candidate_swaps(artifact)

    findings = report["cases"][2]["findings"]
    assert "ATTRIBUTION_ANTISYMMETRY_FAILED" in findings
    assert "ANTI_COSINE_FAILED" in findings


def test_top_k_overlap_detects_salience_reordering() -> None:
    artifact = _artifact()
    artifact["cases"][0]["swapped"]["attributions"] = [
        {"feature_id": "prompt:0", "value": -0.01},
        {"feature_id": "prompt:1", "value": 0.02},
        {"feature_id": "prompt:2", "value": -0.03},
        {"feature_id": "prompt:3", "value": -0.9},
    ]
    artifact["cases"][0]["forward"]["attributions"].append({"feature_id": "prompt:3", "value": 0.001})

    report = audit_candidate_swaps(
        artifact,
        CandidateSwapPolicy(min_features_per_case=3, top_k=2, min_top_k_overlap=1.0),
    )

    assert "TOP_K_OVERLAP_FAILED" in report["cases"][2]["findings"]


def test_weak_margin_is_a_policy_finding() -> None:
    artifact = _artifact()
    artifact["cases"][0]["forward"]["margin"] = 1e-9
    artifact["cases"][0]["swapped"]["margin"] = -1e-9

    report = audit_candidate_swaps(artifact)

    assert report["cases"][2]["findings"] == ["WEAK_PAIRWISE_MARGIN"]


def test_configurable_aggregate_pass_rate_can_admit_one_failed_case() -> None:
    artifact = _artifact()
    artifact["cases"][0]["swapped"]["margin"] = 1.2

    report = audit_candidate_swaps(artifact, CandidateSwapPolicy(min_case_pass_rate=2 / 3))

    assert report["accepted"] is True
    assert report["summary"]["failed_cases"] == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda data: data["cases"][0]["swapped"].update(candidate_order=["candidate-a", "candidate-b"]),
            "candidate_order",
        ),
        (
            lambda data: data["cases"][0]["swapped"]["attributions"][0].update(feature_id="unexpected"),
            "feature IDs do not align",
        ),
        (
            lambda data: data["cases"][0]["forward"]["attributions"].append(
                {"feature_id": "prompt:0", "value": 1.0}
            ),
            "duplicate feature_id",
        ),
        (lambda data: data["cases"][0]["forward"].update(margin=float("nan")), "finite number"),
        (lambda data: data["cases"].append(deepcopy(data["cases"][0])), "duplicate case_id"),
    ],
)
def test_malformed_evidence_fails_closed(mutation, message: str) -> None:
    artifact = _artifact()
    mutation(artifact)

    with pytest.raises(CandidateSwapInputError, match=message):
        audit_candidate_swaps(artifact)


def test_zero_attribution_vector_fails_closed() -> None:
    artifact = _artifact()
    for side in ("forward", "swapped"):
        for item in artifact["cases"][0][side]["attributions"]:
            item["value"] = 0.0

    with pytest.raises(CandidateSwapInputError, match="non-zero L1 norm"):
        audit_candidate_swaps(artifact)


@pytest.mark.parametrize(
    "policy",
    [
        CandidateSwapPolicy(top_k=4, min_features_per_case=3),
        CandidateSwapPolicy(min_case_pass_rate=1.1),
        CandidateSwapPolicy(max_margin_antisymmetry_error=-0.1),
        CandidateSwapPolicy(max_cases=2, min_cases=3),
    ],
)
def test_invalid_policy_fails_closed(policy: CandidateSwapPolicy) -> None:
    with pytest.raises(CandidateSwapInputError, match="policy"):
        audit_candidate_swaps(_artifact(), policy)


def test_digest_is_invariant_to_case_and_feature_order() -> None:
    left = _artifact()
    right = deepcopy(left)
    right["cases"].reverse()
    for case in right["cases"]:
        case["forward"]["attributions"].reverse()
        case["swapped"]["attributions"].reverse()

    left_report = audit_candidate_swaps(left)
    right_report = audit_candidate_swaps(right)

    assert left_report["evidence"]["artifact_sha256"] == right_report["evidence"]["artifact_sha256"]
    assert left_report["cases"] == right_report["cases"]


def test_loader_rejects_duplicate_json_fields(tmp_path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":1,"cases":[]}', encoding="utf-8")

    with pytest.raises(CandidateSwapInputError, match="duplicate field"):
        load_artifact(path)


def test_cli_distinguishes_acceptance_policy_rejection_and_invalid_input(tmp_path) -> None:
    accepted_path = tmp_path / "accepted.json"
    accepted_path.write_text(json.dumps(_artifact()), encoding="utf-8")
    accepted = subprocess.run(
        [sys.executable, "-m", "structxai.candidate_swap", str(accepted_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert accepted.returncode == 0
    assert json.loads(accepted.stdout)["accepted"] is True

    rejected_artifact = _artifact()
    rejected_artifact["cases"][0]["swapped"]["margin"] = 1.2
    rejected_path = tmp_path / "rejected.json"
    rejected_path.write_text(json.dumps(rejected_artifact), encoding="utf-8")
    output_path = tmp_path / "report.json"
    rejected = subprocess.run(
        [
            sys.executable,
            "-m",
            "structxai.candidate_swap",
            str(rejected_path),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 2
    assert json.loads(output_path.read_text(encoding="utf-8"))["accepted"] is False

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{}", encoding="utf-8")
    invalid = subprocess.run(
        [sys.executable, "-m", "structxai.candidate_swap", str(invalid_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert invalid.returncode == 3
    assert json.loads(invalid.stderr)["error_code"] == "INVALID_SWAP_EVIDENCE"


def test_total_feature_budget_is_enforced() -> None:
    with pytest.raises(CandidateSwapInputError, match="max_total_features"):
        audit_candidate_swaps(_artifact(), CandidateSwapPolicy(max_total_features=8))
