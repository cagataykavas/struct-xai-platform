import json
import math

import pytest

from structxai.faithfulness import DeletionStep, evaluate_deletion_curve


def test_progressive_deletion_curve_reports_faithful_effect() -> None:
    steps = [
        DeletionStep(0.0, 4.0),
        DeletionStep(0.25, 3.0),
        DeletionStep(0.5, 2.0),
        DeletionStep(0.75, 1.0),
        DeletionStep(1.0, -1.0),
    ]

    report = evaluate_deletion_curve(
        steps,
        minimum_aopc=2.0,
        minimum_monotonicity=1.0,
    )

    assert report.passed is True
    assert report.aopc == pytest.approx(2.125)
    assert report.monotonicity == pytest.approx(1.0)
    assert report.final_margin_drop == pytest.approx(5.0)
    assert report.peak_margin_drop == pytest.approx(5.0)
    assert report.first_winner_flip_fraction == pytest.approx(1.0)


def test_mean_drop_can_fail_when_deletions_recover_margin() -> None:
    steps = [
        DeletionStep(0.0, 4.0),
        DeletionStep(0.5, 5.0),
        DeletionStep(1.0, 4.0),
    ]

    report = evaluate_deletion_curve(steps, minimum_aopc=0.1)

    assert report.passed is False
    assert report.aopc == pytest.approx(-0.5)
    assert report.monotonicity == pytest.approx(0.5)
    assert report.reasons == ("insufficient_aopc", "non_monotonic_deletion_curve")


def test_small_numeric_recovery_can_be_tolerated_explicitly() -> None:
    steps = [
        DeletionStep(0.0, 2.0),
        DeletionStep(0.5, 1.0),
        DeletionStep(1.0, 1.000001),
    ]

    report = evaluate_deletion_curve(
        steps,
        minimum_monotonicity=1.0,
        recovery_tolerance=0.00001,
    )

    assert report.passed is True
    assert report.monotonicity == pytest.approx(1.0)


def test_report_is_deterministic_and_json_ready() -> None:
    steps = [
        DeletionStep(0.0, 2.0),
        DeletionStep(0.4, 1.0),
        DeletionStep(1.0, -0.5),
    ]

    first = evaluate_deletion_curve(steps).as_dict()
    second = evaluate_deletion_curve(steps).as_dict()

    assert first == second
    assert json.loads(json.dumps(first))["first_winner_flip_fraction"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("steps", "error", "message"),
    [
        ([DeletionStep(0.0, 1.0), DeletionStep(1.0, 0.0)], ValueError, "three"),
        (
            [
                DeletionStep(0.1, 1.0),
                DeletionStep(0.5, 0.5),
                DeletionStep(1.0, 0.0),
            ],
            ValueError,
            "start",
        ),
        (
            [
                DeletionStep(0.0, -1.0),
                DeletionStep(0.5, -2.0),
                DeletionStep(1.0, -3.0),
            ],
            ValueError,
            "positive",
        ),
        (
            [
                DeletionStep(0.0, 1.0),
                DeletionStep(0.5, 0.5),
                DeletionStep(0.5, 0.0),
            ],
            ValueError,
            "strictly increasing",
        ),
        (
            [
                DeletionStep(0.0, 1.0),
                DeletionStep(0.5, math.nan),
                DeletionStep(1.0, 0.0),
            ],
            ValueError,
            "finite",
        ),
    ],
)
def test_invalid_curves_fail_closed(
    steps: list[DeletionStep], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        evaluate_deletion_curve(steps)


def test_policy_configuration_is_validated() -> None:
    steps = [
        DeletionStep(0.0, 1.0),
        DeletionStep(0.5, 0.5),
        DeletionStep(1.0, 0.0),
    ]

    with pytest.raises(ValueError, match="minimum_aopc"):
        evaluate_deletion_curve(steps, minimum_aopc=math.inf)
    with pytest.raises(ValueError, match="minimum_monotonicity"):
        evaluate_deletion_curve(steps, minimum_monotonicity=1.1)
    with pytest.raises(ValueError, match="recovery_tolerance"):
        evaluate_deletion_curve(steps, recovery_tolerance=-0.1)
