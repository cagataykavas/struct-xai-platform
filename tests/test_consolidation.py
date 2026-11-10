from __future__ import annotations

import json

import pytest

from structxai.cli import main
from structxai.consolidation import ConsolidationPolicy, audit_decision_consolidation


def analysis(margins: list[float]) -> dict:
    return {
        "layers": [
            {
                "layer": index,
                "candidate_scores": {"A": margin, "B": 0.0},
            }
            for index, margin in enumerate(margins)
        ]
    }


def test_sustained_pairwise_preference_passes() -> None:
    report = audit_decision_consolidation(analysis([0.1, 0.2, 0.4, 0.8]), ["A", "B"])

    payload = report.to_dict()
    assert payload["schema_version"] == 1
    assert payload["decision"] == "pass"
    assert payload["pairs"][0]["stable_from_layer"] == 0
    assert payload["pairs"][0]["stable_layer_fraction"] == 1.0


def test_late_flip_fails_stable_suffix_policy() -> None:
    report = audit_decision_consolidation(
        analysis([-1.0, -1.0, -1.0, 1.0]),
        ["A", "B"],
        policy=ConsolidationPolicy(min_stable_layer_fraction=0.5),
    )

    assert report.decision == "fail"
    assert report.pairs[0].stable_from_layer == 3
    assert report.pairs[0].last_flip_layer == 3
    assert [item["code"] for item in report.violations] == ["stable_layer_fraction_below_minimum"]


def test_decision_churn_and_final_tie_have_separate_failures() -> None:
    churn = audit_decision_consolidation(
        analysis([1.0, -1.0, 1.0, -1.0, 1.0]),
        ["A", "B"],
        policy=ConsolidationPolicy(max_sign_flips=2),
    )
    assert churn.pairs[0].sign_flip_count == 4
    assert "sign_flip_count_above_maximum" in {item["code"] for item in churn.violations}

    tied = audit_decision_consolidation(analysis([1.0, 0.5, 0.0]), ["A", "B"])
    assert tied.pairs[0].final_winner is None
    assert {item["code"] for item in tied.violations} == {
        "final_margin_below_minimum",
        "stable_layer_fraction_below_minimum",
    }


def test_all_candidate_pairs_are_audited() -> None:
    payload = {
        "layers": [
            {"layer": 0, "candidate_scores": {"A": 3.0, "B": 2.0, "C": 1.0}},
            {"layer": 1, "candidate_scores": {"A": 4.0, "B": 2.0, "C": 0.0}},
        ]
    }
    report = audit_decision_consolidation(payload, ["A", "B", "C"])

    assert report.decision == "pass"
    assert len(report.pairs) == 3


@pytest.mark.parametrize(
    ("payload", "labels", "message"),
    [
        ({"layers": []}, ["A", "B"], "at least two layer rows"),
        (analysis([1.0, 2.0]), ["A", "A"], "must be unique"),
        (
            {
                "layers": [
                    {"layer": 1, "candidate_scores": {"A": 1.0, "B": 0.0}},
                    {"layer": 1, "candidate_scores": {"A": 2.0, "B": 0.0}},
                ]
            },
            ["A", "B"],
            "strictly increasing",
        ),
        (
            {
                "layers": [
                    {"layer": 0, "candidate_scores": {"A": 1.0}},
                    {"layer": 1, "candidate_scores": {"A": 2.0}},
                ]
            },
            ["A", "B"],
            "must be finite",
        ),
        (
            {
                "layers": [
                    {"layer": 0, "candidate_scores": {"A": float("nan"), "B": 0.0}},
                    {"layer": 1, "candidate_scores": {"A": 2.0, "B": 0.0}},
                ]
            },
            ["A", "B"],
            "must be finite",
        ),
    ],
)
def test_malformed_analysis_fails_closed(payload, labels, message) -> None:
    with pytest.raises(ValueError, match=message):
        audit_decision_consolidation(payload, labels)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_stable_layer_fraction": 1.1}, "min_stable_layer_fraction"),
        ({"max_sign_flips": True}, "max_sign_flips"),
        ({"min_final_abs_margin": float("nan")}, "min_final_abs_margin"),
        ({"epsilon": 0}, "epsilon"),
        ({"max_layers": 0}, "max_layers"),
        ({"max_candidate_pairs": -1}, "max_candidate_pairs"),
    ],
)
def test_policy_rejects_invalid_values(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        ConsolidationPolicy(**kwargs)


def test_cli_can_fail_on_unconsolidated_decision(tmp_path) -> None:
    artifact = {
        "experiment": {
            "name": "late-decision",
            "prompt": "Question?",
            "candidates": ["A", "B"],
            "model_name": "tiny",
        },
        "provenance": {},
        "base": analysis([-1.0, -1.0, -1.0, -1.0, 1.0]),
        "variants": {},
    }
    artifact_path = tmp_path / "artifact.json"
    output_path = tmp_path / "evaluation.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    exit_code = main(
        [
            "evaluate",
            str(artifact_path),
            "--output",
            str(output_path),
            "--fail-consolidation",
        ]
    )

    assert exit_code == 3
    assert json.loads(output_path.read_text())["decision_consolidation"]["decision"] == "fail"
