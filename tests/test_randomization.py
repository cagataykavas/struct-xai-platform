import json
import math

import pytest

from structxai.randomization import (
    AttributionCase,
    RandomizationPolicy,
    RandomizationStage,
    evaluate_randomization_sanity,
    main,
)


def case(case_id: str, scores: tuple[float, ...]) -> AttributionCase:
    return AttributionCase(case_id, ("a", "b", "c", "d"), scores)


def stage(stage_id: str, fraction: float, rows: tuple[AttributionCase, ...]) -> RandomizationStage:
    return RandomizationStage(stage_id, fraction, rows)


def policy(**overrides: object) -> RandomizationPolicy:
    values = {
        "top_k": 2,
        "min_cases": 2,
        "min_stages": 2,
        "min_final_randomized_fraction": 0.9,
        "max_final_abs_cosine": 0.4,
        "max_final_top_k_overlap": 0.4,
        "max_final_sign_agreement": 0.6,
        "max_stage_cosine_recovery": 0.05,
    }
    values.update(overrides)
    return RandomizationPolicy(**values)


def baseline() -> tuple[AttributionCase, ...]:
    return (case("q2", (4.0, 3.0, 0.2, 0.1)), case("q1", (3.0, 2.0, 0.2, 0.1)))


def test_progressive_model_sensitivity_passes() -> None:
    stages = (
        stage(
            "top-block",
            0.5,
            (case("q1", (2.0, 1.0, 1.0, 0.5)), case("q2", (2.0, 1.0, 1.0, 0.5))),
        ),
        stage(
            "all-blocks",
            1.0,
            (case("q1", (0.0, 0.0, -3.0, -2.0)), case("q2", (0.0, 0.0, -2.0, -3.0))),
        ),
    )

    report = evaluate_randomization_sanity(baseline(), stages, policy())

    assert report.passed
    assert report.reasons == ()
    assert [row.case_id for row in report.final_case_evidence] == ["q1", "q2"]
    assert report.to_dict()["stages"][-1]["randomized_fraction"] == 1.0


def test_unchanged_explanations_fail_all_final_sensitivity_limits() -> None:
    rows = baseline()
    report = evaluate_randomization_sanity(
        rows,
        (stage("half", 0.5, rows), stage("full", 1.0, rows)),
        policy(),
    )

    assert not report.passed
    assert "final_cosine_too_high" in report.reasons
    assert "final_top_k_overlap_too_high" in report.reasons
    assert "final_sign_agreement_too_high" in report.reasons


def test_similarity_recovery_is_reported() -> None:
    low_similarity = (case("q1", (0.0, 0.0, 3.0, 2.0)), case("q2", (0.0, 0.0, 2.0, 3.0)))
    recovered = (case("q1", (2.8, 1.8, 0.1, 0.1)), case("q2", (3.8, 2.8, 0.1, 0.1)))

    report = evaluate_randomization_sanity(
        baseline(),
        (stage("half", 0.5, low_similarity), stage("full", 1.0, recovered)),
        policy(),
    )

    assert "cosine_recovery_exceeded" in report.reasons


def test_evidence_minimums_fail_closed() -> None:
    rows = (case("q1", (3.0, 2.0, 0.2, 0.1)),)
    report = evaluate_randomization_sanity(
        rows,
        (stage("partial", 0.5, (case("q1", (0.0, 0.0, 2.0, 1.0)),)),),
        policy(),
    )

    assert report.reasons[:3] == (
        "insufficient_cases",
        "insufficient_randomization_stages",
        "insufficient_final_randomization",
    )


def test_stage_case_and_feature_alignment_is_required() -> None:
    with pytest.raises(ValueError, match="baseline case ids"):
        evaluate_randomization_sanity(
            baseline(),
            (stage("full", 1.0, (case("other", (1.0, 2.0, 3.0, 4.0)),)),),
            policy(min_stages=1),
        )

    changed_features = AttributionCase("q1", ("x", "b", "c", "d"), (1.0, 2.0, 3.0, 4.0))
    with pytest.raises(ValueError, match="feature alignment"):
        evaluate_randomization_sanity(
            (baseline()[1],),
            (stage("full", 1.0, (changed_features,)),),
            policy(min_cases=1, min_stages=1),
        )


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (AttributionCase("q1", ("a", "a"), (1.0, 2.0)), "invalid feature ids"),
        (AttributionCase("q1", ("a", "b"), (0.0, 0.0)), "zero attribution"),
        (AttributionCase("q1", ("a", "b"), (1.0, math.nan)), "non-finite"),
        (AttributionCase("q1", ("a",), (1.0, 2.0)), "misaligned"),
    ],
)
def test_malformed_attribution_vectors_are_rejected(row: AttributionCase, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_randomization_sanity(
            (row,),
            (stage("full", 1.0, (row,)),),
            policy(min_cases=1, min_stages=1),
        )


def test_stage_ids_and_fractions_must_be_ordered_and_unique() -> None:
    rows = baseline()
    with pytest.raises(ValueError, match="strictly increasing"):
        evaluate_randomization_sanity(
            rows,
            (stage("full", 1.0, rows), stage("partial", 0.5, rows)),
            policy(),
        )
    with pytest.raises(ValueError, match="duplicate stage_id"):
        evaluate_randomization_sanity(
            rows,
            (stage("same", 0.5, rows), stage("same", 1.0, rows)),
            policy(),
        )


def test_duplicate_case_ids_are_rejected() -> None:
    row = case("q1", (1.0, 2.0, 3.0, 4.0))
    with pytest.raises(ValueError, match="duplicate baseline case_id"):
        evaluate_randomization_sanity((row, row), (stage("full", 1.0, (row,)),), policy())


def test_invalid_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_final_abs_cosine"):
        policy(max_final_abs_cosine=1.1)


def test_cli_distinguishes_policy_rejection_from_bad_input(tmp_path, capsys) -> None:
    artifact = tmp_path / "randomization.json"
    row = {
        "case_id": "q1",
        "feature_ids": ["a", "b"],
        "scores": [2.0, 1.0],
    }
    artifact.write_text(
        json.dumps(
            {
                "policy": {"top_k": 1, "min_cases": 1, "min_stages": 1},
                "baseline": [row],
                "stages": [{"stage_id": "full", "randomized_fraction": 1.0, "cases": [row]}],
            }
        ),
        encoding="utf-8",
    )

    assert main([str(artifact), "--require-pass"]) == 2
    assert json.loads(capsys.readouterr().out)["passed"] is False

    artifact.write_text("not-json", encoding="utf-8")
    assert main([str(artifact), "--require-pass"]) == 1
    assert "error" in json.loads(capsys.readouterr().out)
