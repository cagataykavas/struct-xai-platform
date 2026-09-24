from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from structxai.patch_controls import (
    PatchControlPolicy,
    audit_patch_controls,
    build_case_from_scores,
)


def _case(case_id: str, *, target: float = 0.8, sham: float = 0.03) -> dict:
    return {
        "case_id": case_id,
        "clean_margin": 2.0,
        "corrupted_margin": 0.0,
        "target_patch": {"site_id": "layer-8:final", "patched_margin": target * 2.0},
        "sham_patch": {"site_id": "embedding:padding", "patched_margin": sham * 2.0},
        "random_controls": [
            {"site_id": f"random-{index}", "patched_margin": effect * 2.0}
            for index, effect in enumerate((0.02, -0.04, 0.06, 0.08, -0.1), start=1)
        ],
    }


def _artifact() -> dict:
    return {"experiment_id": "ioi-control-v1", "cases": [_case(f"case-{index}") for index in range(3)]}


def test_accepts_specific_target_recovery() -> None:
    report = audit_patch_controls(_artifact())
    assert report.accepted is True
    assert report.malformed is False
    assert report.experiment_id == "ioi-control-v1"
    assert report.case_count == 3
    assert report.case_pass_rate == 1.0
    assert report.mean_target_recovery_fraction == pytest.approx(0.8)
    assert report.cases[0].control_p95_effect_fraction == pytest.approx(0.096)
    assert report.cases[0].specificity_gap == pytest.approx(0.704)


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda value: value["cases"][0].update(clean_margin=0.1),
            "INSUFFICIENT_CORRUPTION_EFFECT",
        ),
        (
            lambda value: value["cases"][0]["target_patch"].update(patched_margin=0.4),
            "TARGET_RECOVERY_BELOW_MINIMUM",
        ),
        (
            lambda value: value["cases"][0]["target_patch"].update(patched_margin=4.0),
            "TARGET_RECOVERY_ABOVE_MAXIMUM",
        ),
        (
            lambda value: value["cases"][0]["sham_patch"].update(patched_margin=0.8),
            "SHAM_EFFECT_ABOVE_MAXIMUM",
        ),
        (
            lambda value: value["cases"][0]["random_controls"][4].update(patched_margin=1.0),
            "CONTROL_EFFECT_ABOVE_MAXIMUM",
        ),
        (
            lambda value: value["cases"][0].update(random_controls=[]),
            "INSUFFICIENT_RANDOM_CONTROLS",
        ),
    ],
)
def test_case_controls_reject_non_specific_effects(mutate, reason: str) -> None:
    artifact = _artifact()
    mutate(artifact)
    report = audit_patch_controls(artifact)
    assert report.accepted is False
    assert report.malformed is False
    assert "CASE_PASS_RATE_BELOW_MINIMUM" in report.reason_codes
    assert reason in report.cases[0].reason_codes


def test_negative_clean_gap_normalizes_recovery_direction() -> None:
    artifact = _artifact()
    for case in artifact["cases"]:
        case["clean_margin"] = -2.0
        case["target_patch"]["patched_margin"] = -1.6
        case["sham_patch"]["patched_margin"] = -0.02
        for control in case["random_controls"]:
            control["patched_margin"] = 0.02
    report = audit_patch_controls(artifact)
    assert report.accepted is True
    assert report.cases[0].target_recovery_fraction == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda value: value["cases"].append(deepcopy(value["cases"][0])), "DUPLICATE_CASE_ID"),
        (
            lambda value: value["cases"][0]["sham_patch"].update(site_id="layer-8:final"),
            "DUPLICATE_PATCH_SITE",
        ),
        (lambda value: value["cases"][0].update(clean_margin=float("nan")), "INVALID_MARGIN"),
        (lambda value: value.update(cases="not-an-array"), "INVALID_CASES"),
        (lambda value: value.pop("experiment_id"), "INVALID_IDENTIFIER"),
    ],
)
def test_malformed_artifacts_fail_closed(mutate, reason: str) -> None:
    artifact = _artifact()
    mutate(artifact)
    report = audit_patch_controls(artifact)
    assert report.accepted is False
    assert report.malformed is True
    assert report.reason_codes == (reason,)
    assert report.error_path


def test_aggregate_case_count_and_pass_rate_are_enforced() -> None:
    artifact = _artifact()
    artifact["cases"] = artifact["cases"][:2]
    report = audit_patch_controls(artifact)
    assert report.reason_codes == ("INSUFFICIENT_CASES",)

    artifact = _artifact()
    artifact["cases"][0]["target_patch"]["patched_margin"] = 0.0
    artifact["cases"][1]["target_patch"]["patched_margin"] = 0.0
    report = audit_patch_controls(artifact)
    assert "CASE_PASS_RATE_BELOW_MINIMUM" in report.reason_codes
    assert report.case_pass_rate == pytest.approx(1 / 3)


def test_policy_can_tolerate_a_bounded_fraction_of_failed_cases() -> None:
    artifact = _artifact()
    artifact["cases"].extend([_case("case-3"), _case("case-4")])
    artifact["cases"][0]["target_patch"]["patched_margin"] = 0.0
    report = audit_patch_controls(artifact)
    assert report.accepted is True
    assert report.case_pass_rate == 0.8


def test_score_builder_uses_existing_patch_score_contract() -> None:
    case = build_case_from_scores(
        case_id="case-1",
        positive_candidate="A",
        negative_candidate="B",
        clean_scores={"A": 2.0, "B": 0.0},
        corrupted_scores={"A": 0.0, "B": 0.0},
        target_site_id="layer-8",
        target_scores={"A": 1.6, "B": 0.0},
        sham_site_id="layer-0",
        sham_scores={"A": 0.05, "B": 0.0},
        random_scores={f"layer-{index}": {"A": 0.02, "B": 0.0} for index in range(1, 6)},
    )
    artifact = {
        "experiment_id": "score-contract-v1",
        "cases": [asdict_case(case), asdict_case(case, "case-2"), asdict_case(case, "case-3")],
    }
    assert audit_patch_controls(artifact).accepted is True


def asdict_case(case, case_id: str | None = None) -> dict:
    return {
        "case_id": case_id or case.case_id,
        "clean_margin": case.clean_margin,
        "corrupted_margin": case.corrupted_margin,
        "target_patch": {
            "site_id": case.target_patch.site_id,
            "patched_margin": case.target_patch.patched_margin,
        },
        "sham_patch": {
            "site_id": case.sham_patch.site_id,
            "patched_margin": case.sham_patch.patched_margin,
        },
        "random_controls": [
            {"site_id": item.site_id, "patched_margin": item.patched_margin} for item in case.random_controls
        ],
    }


def test_invalid_policy_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        PatchControlPolicy(min_cases=True)
    with pytest.raises(ValueError, match="below the minimum"):
        PatchControlPolicy(
            min_target_recovery_fraction=1.0,
            max_target_recovery_fraction=0.5,
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        PatchControlPolicy(min_case_pass_rate=1.1)


def test_cli_exit_codes_distinguish_accept_reject_and_malformed(tmp_path: Path) -> None:
    path = tmp_path / "controls.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")
    command = [sys.executable, "-m", "structxai.patch_controls", str(path)]

    accepted = subprocess.run(command, capture_output=True, text=True, check=False)
    assert accepted.returncode == 0
    assert json.loads(accepted.stdout)["accepted"] is True

    rejected = subprocess.run(
        [*command, "--min-target-recovery", "0.95"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode == 3
    assert json.loads(rejected.stdout)["malformed"] is False

    path.write_text("{broken", encoding="utf-8")
    malformed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert malformed.returncode == 2
    assert json.loads(malformed.stdout)["malformed"] is True
