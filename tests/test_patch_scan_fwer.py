from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime

import pytest

from structxai.patch_scan_fwer import (
    ArtifactMalformed,
    ScanPolicy,
    audit_patch_scan,
    load_artifact,
    main,
)

AS_OF = datetime(2026, 9, 27, 1, 0, tzinfo=UTC)
DIGEST = "a" * 64


def _sites(values: list[float]) -> list[dict[str, object]]:
    return [{"site_id": f"layer.{index}", "patched_margin": value} for index, value in enumerate(values)]


def _case(
    case_id: str,
    *,
    winning_site: int = 0,
    clean_margin: float = 1.0,
    corrupted_margin: float = 0.0,
    null_replicates: int = 19,
    null_values: list[float] | None = None,
) -> dict[str, object]:
    direction = 1.0 if clean_margin >= corrupted_margin else -1.0
    observed = [corrupted_margin + direction * 0.10 for _ in range(4)]
    observed[winning_site] = corrupted_margin + direction * 0.80
    null = null_values or [corrupted_margin + direction * 0.10 for _ in range(4)]
    return {
        "case_id": case_id,
        "clean_margin": clean_margin,
        "corrupted_margin": corrupted_margin,
        "sites": _sites(observed),
        "null_replicates": [
            {"replicate_id": f"null-{index}", "sites": _sites(null)} for index in range(null_replicates)
        ],
    }


def _artifact(cases: list[dict[str, object]] | None = None) -> dict[str, object]:
    selected_cases = cases or [_case("case-a"), _case("case-b"), _case("case-c")]
    site_ids = sorted(site["site_id"] for site in selected_cases[0]["sites"])
    site_manifest = json.dumps(site_ids, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": 1,
        "experiment_id": "scan-2026-09-27",
        "model_sha256": DIGEST,
        "patch_config_sha256": "b" * 64,
        "eligible_sites_sha256": hashlib.sha256(site_manifest).hexdigest(),
        "created_at": "2026-09-27T00:59:30Z",
        "cases": selected_cases,
    }


def _codes(report) -> set[str]:
    return {finding.code for finding in report.findings}


def test_accepts_recurrent_site_with_fwer_control() -> None:
    report = audit_patch_scan(_artifact(), as_of=AS_OF)

    assert report.accepted is True
    assert report.status == "accepted"
    assert report.case_count == 3
    assert report.site_count == 4
    assert report.null_replicates_per_case == 19
    assert report.total_patch_values == 240
    assert report.significant_case_count == 3
    assert report.significant_case_fraction == 1.0
    assert report.recurrent_site_count == 1
    assert report.cases[0].minimum_corrected_p == pytest.approx(0.05)
    assert report.recurrent_sites[0].significant_case_fraction == 1.0


def test_directional_recovery_supports_negative_clean_margin_gap() -> None:
    cases = [
        _case(case_id, clean_margin=-1.0, corrupted_margin=0.0) for case_id in ("case-a", "case-b", "case-c")
    ]

    report = audit_patch_scan(_artifact(cases), as_of=AS_OF)

    assert report.accepted is True
    assert all(case.peak_recovery == pytest.approx(0.8) for case in report.cases)


def test_max_statistic_uses_largest_null_effect_across_sites() -> None:
    artifact = _artifact()
    for case in artifact["cases"]:
        case["null_replicates"][0]["sites"][3]["patched_margin"] = 0.90

    report = audit_patch_scan(artifact, as_of=AS_OF)

    assert report.accepted is False
    assert report.significant_case_count == 0
    assert all(case.minimum_corrected_p == pytest.approx(0.10) for case in report.cases)
    assert "NO_FWER_SIGNIFICANT_SITE" in _codes(report)


def test_recovery_below_minimum_is_not_significant() -> None:
    artifact = _artifact()
    for case in artifact["cases"]:
        case["sites"][0]["patched_margin"] = 0.19

    report = audit_patch_scan(artifact, as_of=AS_OF)

    assert report.significant_case_count == 0
    assert "NO_FWER_SIGNIFICANT_SITE" in _codes(report)


def test_recovery_overshoot_is_reported_and_not_significant() -> None:
    artifact = _artifact()
    for case in artifact["cases"]:
        case["sites"][0]["patched_margin"] = 1.51

    report = audit_patch_scan(artifact, as_of=AS_OF)

    assert "PATCH_RECOVERY_OVERSHOOT" in _codes(report)
    assert report.significant_case_count == 0


def test_small_corruption_gap_fails_closed() -> None:
    cases = [
        _case(case_id, clean_margin=0.09, corrupted_margin=0.0) for case_id in ("case-a", "case-b", "case-c")
    ]

    report = audit_patch_scan(_artifact(cases), as_of=AS_OF)

    assert report.accepted is False
    assert "CORRUPTION_GAP_BELOW_POLICY" in _codes(report)
    assert all(case.minimum_corrected_p is None for case in report.cases)


def test_significant_case_fraction_is_an_aggregate_gate() -> None:
    artifact = _artifact()
    for case in artifact["cases"][1:]:
        case["sites"][0]["patched_margin"] = 0.10

    report = audit_patch_scan(artifact, as_of=AS_OF)

    assert report.significant_case_count == 1
    assert report.significant_case_fraction == pytest.approx(1 / 3)
    assert "SIGNIFICANT_CASE_FRACTION_BELOW_POLICY" in _codes(report)


def test_recurrence_requires_same_site_across_cases() -> None:
    cases = [
        _case("case-a", winning_site=0),
        _case("case-b", winning_site=1),
        _case("case-c", winning_site=2),
    ]
    policy = ScanPolicy(min_recurrent_site_fraction=0.67)

    report = audit_patch_scan(_artifact(cases), as_of=AS_OF, policy=policy)

    assert report.significant_case_count == 3
    assert report.recurrent_site_count == 0
    assert "NO_RECURRENT_SIGNIFICANT_SITE" in _codes(report)


@pytest.mark.parametrize(
    ("cases", "policy", "expected_code"),
    [
        ([_case("case-a")], ScanPolicy(), "CASE_EVIDENCE_UNDERPOWERED"),
        (
            [_case("case-a"), _case("case-b"), _case("case-c")],
            ScanPolicy(min_sites=5),
            "SITE_EVIDENCE_UNDERPOWERED",
        ),
        (
            [
                _case("case-a", null_replicates=10),
                _case("case-b", null_replicates=10),
                _case("case-c", null_replicates=10),
            ],
            ScanPolicy(),
            "NULL_EVIDENCE_UNDERPOWERED",
        ),
    ],
)
def test_underpowered_evidence_is_rejected(cases, policy, expected_code) -> None:
    report = audit_patch_scan(_artifact(cases), as_of=AS_OF, policy=policy)

    assert report.accepted is False
    assert expected_code in _codes(report)


def test_policy_rejects_unresolvable_alpha() -> None:
    with pytest.raises(ValueError, match="cannot resolve"):
        ScanPolicy(min_null_replicates=9, family_wise_alpha=0.05)


def test_case_site_sets_must_align() -> None:
    artifact = _artifact()
    artifact["cases"][1]["sites"][0]["site_id"] = "other-site"
    for replicate in artifact["cases"][1]["null_replicates"]:
        replicate["sites"][0]["site_id"] = "other-site"

    with pytest.raises(ArtifactMalformed, match="CASE_SITE_SET_MISMATCH"):
        audit_patch_scan(artifact, as_of=AS_OF)


def test_null_site_sets_must_align() -> None:
    artifact = _artifact()
    artifact["cases"][0]["null_replicates"][0]["sites"][0]["site_id"] = "other-site"

    with pytest.raises(ArtifactMalformed, match="NULL_SITE_SET_MISMATCH"):
        audit_patch_scan(artifact, as_of=AS_OF)


def test_null_replicate_counts_must_align() -> None:
    artifact = _artifact()
    artifact["cases"][0]["null_replicates"].pop()

    with pytest.raises(ArtifactMalformed, match="NULL_REPLICATE_COUNT_MISMATCH"):
        audit_patch_scan(artifact, as_of=AS_OF)


def test_eligible_site_manifest_digest_must_match_sites() -> None:
    artifact = _artifact()
    artifact["eligible_sites_sha256"] = "d" * 64

    with pytest.raises(ArtifactMalformed, match="ELIGIBLE_SITES_DIGEST_MISMATCH"):
        audit_patch_scan(artifact, as_of=AS_OF)


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda artifact: artifact["cases"].append(copy.deepcopy(artifact["cases"][0])), "DUPLICATE_CASE_ID"),
        (
            lambda artifact: artifact["cases"][0]["sites"].append(
                copy.deepcopy(artifact["cases"][0]["sites"][0])
            ),
            "DUPLICATE_SITE_ID",
        ),
        (
            lambda artifact: artifact["cases"][0]["null_replicates"].append(
                copy.deepcopy(artifact["cases"][0]["null_replicates"][0])
            ),
            "DUPLICATE_REPLICATE_ID",
        ),
    ],
)
def test_duplicate_identifiers_are_malformed(mutate, code) -> None:
    artifact = _artifact()
    mutate(artifact)

    with pytest.raises(ArtifactMalformed, match=code):
        audit_patch_scan(artifact, as_of=AS_OF)


@pytest.mark.parametrize("bad_value", [True, "0.8", 1_000_001.0, float("inf")])
def test_patch_values_are_finite_bounded_numbers(bad_value) -> None:
    artifact = _artifact()
    artifact["cases"][0]["sites"][0]["patched_margin"] = bad_value

    with pytest.raises(ArtifactMalformed, match="PATCHED_MARGIN"):
        audit_patch_scan(artifact, as_of=AS_OF)


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        (b'{"schema_version":1,"schema_version":1}', "DUPLICATE_JSON_KEY"),
        (b'{"value":NaN}', "NON_FINITE_NUMBER"),
        (b"\xff", "INVALID_UTF8"),
        (b"[]", "ROOT_NOT_OBJECT"),
    ],
)
def test_strict_json_loader(raw: bytes, code: str) -> None:
    with pytest.raises(ArtifactMalformed, match=code):
        load_artifact(raw)


def test_loader_enforces_byte_budget() -> None:
    with pytest.raises(ArtifactMalformed, match="INPUT_TOO_LARGE"):
        load_artifact(b"{} ", ScanPolicy(max_input_bytes=2))


@pytest.mark.parametrize(
    ("created_at", "code"),
    [
        ("2026-09-25T00:00:00Z", "STALE_EVIDENCE"),
        ("2026-09-27T01:02:00Z", "EVIDENCE_FROM_FUTURE"),
        ("2026-09-27T00:59:30+00:00", "CREATED_AT"),
    ],
)
def test_evidence_timestamp_is_bounded(created_at: str, code: str) -> None:
    artifact = _artifact()
    artifact["created_at"] = created_at

    with pytest.raises(ArtifactMalformed, match=code):
        audit_patch_scan(artifact, as_of=AS_OF)


@pytest.mark.parametrize(
    "target",
    [
        lambda artifact: artifact.update({"unexpected": True}),
        lambda artifact: artifact["cases"][0].update({"unexpected": True}),
        lambda artifact: artifact["cases"][0]["sites"][0].update({"unexpected": True}),
        lambda artifact: artifact["cases"][0]["null_replicates"][0].update({"unexpected": True}),
    ],
)
def test_unknown_fields_are_rejected(target) -> None:
    artifact = _artifact()
    target(artifact)

    with pytest.raises(ArtifactMalformed):
        audit_patch_scan(artifact, as_of=AS_OF)


@pytest.mark.parametrize(
    ("policy", "code"),
    [
        (ScanPolicy(max_cases=2, min_cases=2), "CASE_BUDGET_EXCEEDED"),
        (ScanPolicy(max_sites=3, min_sites=3), "SITE_BUDGET_EXCEEDED"),
        (
            ScanPolicy(
                max_null_replicates=18,
                min_null_replicates=18,
                family_wise_alpha=0.10,
            ),
            "NULL_REPLICATE_BUDGET_EXCEEDED",
        ),
        (ScanPolicy(max_total_patch_values=239), "PATCH_VALUE_BUDGET_EXCEEDED"),
    ],
)
def test_resource_budgets_fail_closed(policy: ScanPolicy, code: str) -> None:
    with pytest.raises(ArtifactMalformed, match=code):
        audit_patch_scan(_artifact(), as_of=AS_OF, policy=policy)


def test_artifact_digest_is_independent_of_collection_order() -> None:
    left = _artifact()
    right = copy.deepcopy(left)
    right["cases"].reverse()
    for case in right["cases"]:
        case["sites"].reverse()
        case["null_replicates"].reverse()
        for replicate in case["null_replicates"]:
            replicate["sites"].reverse()

    left_report = audit_patch_scan(left, as_of=AS_OF)
    right_report = audit_patch_scan(right, as_of=AS_OF)

    assert left_report.artifact_sha256 == right_report.artifact_sha256
    assert left_report.to_dict() == right_report.to_dict()


def test_report_does_not_disclose_raw_identifiers_or_margins() -> None:
    artifact = _artifact()
    serialized = json.dumps(audit_patch_scan(artifact, as_of=AS_OF).to_dict())

    assert artifact["experiment_id"] not in serialized
    assert all(case["case_id"] not in serialized for case in artifact["cases"])
    assert all(site["site_id"] not in serialized for site in artifact["cases"][0]["sites"])
    assert "clean_margin" not in serialized
    assert "corrupted_margin" not in serialized
    assert "patched_margin" not in serialized


def test_findings_are_bounded_but_count_is_preserved() -> None:
    artifact = _artifact()
    for case in artifact["cases"]:
        case["sites"][0]["patched_margin"] = 1.60
    policy = ScanPolicy(max_findings=2)

    report = audit_patch_scan(artifact, as_of=AS_OF, policy=policy)

    assert report.findings_truncated is True
    assert report.finding_count > 2
    assert len(report.findings) == 2


def test_policy_digest_changes_with_threshold() -> None:
    default = audit_patch_scan(_artifact(), as_of=AS_OF)
    stricter = audit_patch_scan(
        _artifact(),
        as_of=AS_OF,
        policy=ScanPolicy(min_recovery=0.30),
    )

    assert default.policy_sha256 != stricter.policy_sha256


def test_cli_writes_atomic_reports_and_uses_stable_exit_codes(tmp_path) -> None:
    artifact_path = tmp_path / "artifact.json"
    output_path = tmp_path / "report.json"
    artifact_path.write_text(json.dumps(_artifact()), encoding="utf-8")

    assert main([str(artifact_path), "--as-of", "2026-09-27T01:00:00Z", "--output", str(output_path)]) == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["accepted"] is True

    rejected = _artifact()
    for case in rejected["cases"]:
        case["sites"][0]["patched_margin"] = 0.10
    artifact_path.write_text(json.dumps(rejected), encoding="utf-8")
    assert main([str(artifact_path), "--as-of", "2026-09-27T01:00:00Z", "--output", str(output_path)]) == 2

    artifact_path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    assert main([str(artifact_path), "--output", str(output_path)]) == 3
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "malformed"
    assert list(tmp_path.glob(".report.json.*")) == []


def test_medium_scan_remains_bounded_and_deterministic() -> None:
    site_count = 32
    replicate_count = 49
    cases = []
    for case_index in range(50):
        observed = [0.05] * site_count
        observed[7] = 0.80
        null = [0.05] * site_count
        cases.append(
            {
                "case_id": f"case-{case_index:03d}",
                "clean_margin": 1.0,
                "corrupted_margin": 0.0,
                "sites": _sites(observed),
                "null_replicates": [
                    {"replicate_id": f"null-{index:03d}", "sites": _sites(null)}
                    for index in range(replicate_count)
                ],
            }
        )
    policy = ScanPolicy(
        min_null_replicates=49,
        family_wise_alpha=0.02,
        max_cases=50,
        max_sites=32,
    )

    report = audit_patch_scan(_artifact(cases), as_of=AS_OF, policy=policy)

    assert report.accepted is True
    assert report.total_patch_values == 50 * 32 * 50
    assert report.recurrent_sites[0].significant_case_count == 50
