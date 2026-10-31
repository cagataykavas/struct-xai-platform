import copy
import json

import pytest

from structxai.stability import StabilityCohortPolicy, audit_artifact_stability_cohort
from structxai.stability_uncertainty import (
    StabilityTaskCohortError,
    StabilityTaskCohortPolicy,
    audit_stability_task_cohort,
    require_stability_task_cohort,
)


def _artifact(margins: list[float], *, task_index: int) -> dict:
    return {
        "experiment": {
            "name": f"task-{task_index}",
            "prompt": f"Private task prompt {task_index}",
            "candidates": ["A", "B"],
            "model_name": "tiny-model",
        },
        "created_at": "2026-10-31T10:00:00+00:00",
        "base": {
            "layers": [
                {
                    "layer": layer,
                    "candidate_scores": {"A": margin, "B": 0.0},
                }
                for layer, margin in enumerate(margins)
            ]
        },
        "provenance": {"experiment_fingerprint": f"{task_index + 1:064x}"},
    }


def _task_report(
    task_index: int,
    *,
    unstable: bool = False,
    run_policy: StabilityCohortPolicy | None = None,
    run_count: int = 3,
) -> dict:
    runs = {
        f"seed-{seed}": _artifact(
            [1.0 + seed / 10_000, 2.0 + seed / 10_000, 3.0 + seed / 10_000],
            task_index=task_index,
        )
        for seed in range(run_count)
    }
    if unstable:
        runs[f"seed-{run_count - 1}"] = _artifact([1.0, 2.0, -1.0], task_index=task_index)
    return audit_artifact_stability_cohort(runs, policy=run_policy)


def _reports(*, unstable_from: int | None = None) -> dict[str, dict]:
    return {
        f"public-task-{index:02d}": _task_report(
            index,
            unstable=unstable_from is not None and index >= unstable_from,
        )
        for index in range(8)
    }


def test_task_cohort_is_deterministic_private_and_accepts_stable_tasks():
    reports = _reports()

    first = audit_stability_task_cohort(reports)
    second = audit_stability_task_cohort(dict(reversed(tuple(reports.items()))))

    assert first == second
    assert first["accepted"] is True
    assert first["task_count"] == 8
    assert first["run_count_per_task"] == 3
    assert first["bootstrap_work_units"] == 16_000
    assert first["finding_codes"] == []
    assert first["resampling_unit"] == "task"
    assert len(first["evidence_sha256"]) == 64
    assert "Private task prompt" not in json.dumps(first)
    assert require_stability_task_cohort(reports) == first


def test_task_cohort_reports_uncertainty_bound_failures():
    report = audit_stability_task_cohort(_reports(unstable_from=4))

    assert report["accepted"] is False
    assert set(report["finding_codes"]) == {
        "ACCEPTED_TASK_RATE_LOWER_BOUND_BELOW_FLOOR",
        "MEAN_TASK_FINAL_MARGIN_RANGE_UPPER_BOUND_ABOVE_LIMIT",
        "MEAN_TASK_MARGIN_DRIFT_UPPER_BOUND_ABOVE_LIMIT",
        "MEAN_TASK_PEARSON_LOWER_BOUND_BELOW_FLOOR",
        "MEAN_TASK_SIGN_AGREEMENT_LOWER_BOUND_BELOW_FLOOR",
    }
    with pytest.raises(StabilityTaskCohortError, match="policy failed"):
        require_stability_task_cohort(_reports(unstable_from=4))


def test_task_cohort_uses_task_level_worst_pair_metrics():
    report = audit_stability_task_cohort(_reports())
    intervals = report["intervals"]

    assert intervals["mean_task_minimum_pairwise_pearson"]["estimate"] == pytest.approx(1.0)
    assert intervals["mean_task_minimum_pairwise_sign_agreement"]["lower"] == pytest.approx(1.0)
    assert intervals["accepted_task_rate"]["estimate"] == pytest.approx(1.0)
    assert 0.0 < intervals["accepted_task_rate"]["lower"] < 1.0


def test_task_cohort_rejects_tampered_report_before_resampling():
    reports = _reports()
    reports["public-task-00"]["accepted"] = False

    with pytest.raises(StabilityTaskCohortError, match="digest does not match"):
        audit_stability_task_cohort(reports)


def test_task_cohort_rejects_duplicate_experiment_commitments():
    reports = _reports()
    reports["public-task-07"] = copy.deepcopy(reports["public-task-06"])

    with pytest.raises(StabilityTaskCohortError, match="duplicate experiment"):
        audit_stability_task_cohort(reports)


def test_task_cohort_requires_one_run_policy_and_run_count():
    reports = _reports()
    reports["public-task-07"] = _task_report(
        7,
        run_policy=StabilityCohortPolicy(min_pairwise_pearson=0.90),
    )
    with pytest.raises(StabilityTaskCohortError, match="different run-cohort policies"):
        audit_stability_task_cohort(reports)

    reports = _reports()
    reports["public-task-07"] = _task_report(7, run_count=4)
    with pytest.raises(StabilityTaskCohortError, match="different run counts"):
        audit_stability_task_cohort(reports)


def test_task_cohort_requires_enough_tasks_and_canonical_ids():
    with pytest.raises(StabilityTaskCohortError, match="task count"):
        audit_stability_task_cohort({"task": _task_report(0)})

    reports = _reports()
    reports["unsafe task id"] = reports.pop("public-task-07")
    with pytest.raises(StabilityTaskCohortError, match="task IDs"):
        audit_stability_task_cohort(reports)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"min_tasks": 7}, "min_tasks"),
        ({"min_tasks": 10, "max_tasks": 9}, "max_tasks"),
        ({"bootstrap_resamples": 499}, "bootstrap_resamples"),
        ({"bootstrap_seed": -1}, "bootstrap_seed"),
        ({"confidence_level": float("nan")}, "finite"),
        ({"min_mean_task_sign_agreement_lower_bound": 1.1}, "between"),
        ({"max_mean_task_margin_drift_upper_bound": -0.1}, "non-negative"),
    ],
)
def test_task_cohort_policy_is_bounded(kwargs, match):
    with pytest.raises(ValueError, match=match):
        StabilityTaskCohortPolicy(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence_level": True},
        {"min_accepted_task_rate_lower_bound": "0.6"},
        {"max_mean_task_final_margin_range_upper_bound": False},
    ],
)
def test_task_cohort_policy_rejects_non_numeric_thresholds(kwargs):
    with pytest.raises(TypeError, match="must be numeric"):
        StabilityTaskCohortPolicy(**kwargs)
