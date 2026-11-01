import copy
import hashlib

import pytest

from structxai.stability import StabilityCohortPolicy, audit_artifact_stability_cohort
from structxai.stability_manifest import (
    StabilityTaskManifestError,
    create_signed_stability_task_evidence,
    create_stability_task_manifest,
    validate_stability_task_manifest,
    verify_signed_stability_task_evidence,
)
from structxai.stability_uncertainty import StabilityTaskCohortError, StabilityTaskCohortPolicy


class _TestExternalAdapter:
    algorithm = "ED25519"
    key_id = "portfolio-evidence-key"
    key_revision_sha256 = "9" * 64

    def __init__(self, secret: bytes = b"test-only-external-key-revision") -> None:
        self._secret = secret
        self.last_message = b""

    def sign(self, message: bytes) -> bytes:
        self.last_message = message
        return hashlib.sha512(self._secret + message).digest()

    def verify(self, message: bytes, signature: bytes) -> bool:
        self.last_message = message
        return signature == hashlib.sha512(self._secret + message).digest()


class _FailingSigner(_TestExternalAdapter):
    def sign(self, message: bytes) -> bytes:
        raise RuntimeError("provider diagnostic must stay private")


def _artifact(task_index: int, seed: int) -> dict:
    return {
        "experiment": {
            "name": f"task-{task_index}",
            "prompt": f"Private public-task rendering {task_index}",
            "candidates": ["A", "B"],
            "model_name": "pinned-model",
        },
        "created_at": "2026-11-01T00:00:00+00:00",
        "base": {
            "layers": [
                {
                    "layer": layer,
                    "candidate_scores": {"A": margin + seed / 10_000, "B": 0.0},
                }
                for layer, margin in enumerate((1.0, 2.0, 3.0))
            ]
        },
        "provenance": {"experiment_fingerprint": f"{task_index + 1:064x}"},
    }


def _reports() -> dict[str, dict]:
    return {
        f"public-task-{task_index:02d}": audit_artifact_stability_cohort(
            {f"seed-{seed}": _artifact(task_index, seed) for seed in range(3)}
        )
        for task_index in range(8)
    }


def _manifest(reports: dict[str, dict] | None = None, **overrides) -> dict:
    active_reports = reports or _reports()
    arguments = {
        "cohort_id": "public-xai-v1",
        "dataset_uri": "https://example.org/datasets/xai-eval",
        "dataset_revision": "1" * 40,
        "dataset_records_sha256": "2" * 64,
        "selection_protocol_uri": "https://example.org/protocols/xai-selection.json",
        "selection_protocol_sha256": "3" * 64,
        "model_uri": "https://example.org/models/pinned-model",
        "model_revision": "4" * 40,
        "model_weights_sha256": "5" * 64,
        "tokenizer_sha256": "6" * 64,
        "run_cohort_policy_sha256": next(iter(active_reports.values()))["policy_sha256"],
        "task_cohort_policy": StabilityTaskCohortPolicy(),
    }
    arguments.update(overrides)
    return create_stability_task_manifest(
        {task_id: report["experiment_sha256"] for task_id, report in active_reports.items()},
        **arguments,
    )


def test_manifest_binds_and_signs_accepted_evidence_without_private_payloads():
    reports = _reports()
    manifest = _manifest(reports)
    adapter = _TestExternalAdapter()

    bundle = create_signed_stability_task_evidence(manifest, reports, signer=adapter)
    verification = verify_signed_stability_task_evidence(
        bundle,
        {adapter.key_id: adapter},
    )

    assert verification["verified"] is True
    assert verification["cohort_id"] == "public-xai-v1"
    assert verification["task_count"] == 8
    assert verification["manifest_sha256"] == manifest["manifest_sha256"]
    assert len(adapter.last_message) < 4_096
    assert b"Private public-task rendering" not in adapter.last_message
    assert "Private public-task rendering" not in str(bundle)


def test_manifest_and_signature_are_deterministic_under_input_reordering():
    reports = _reports()
    adapter = _TestExternalAdapter()

    first = create_signed_stability_task_evidence(_manifest(reports), reports, signer=adapter)
    second = create_signed_stability_task_evidence(
        _manifest(dict(reversed(tuple(reports.items())))),
        dict(reversed(tuple(reports.items()))),
        signer=adapter,
    )

    assert first == second


def test_manifest_rejects_post_selected_or_substituted_task():
    reports = _reports()
    manifest = _manifest(reports)
    substituted = _reports()
    substituted["public-task-07"] = audit_artifact_stability_cohort(
        {f"seed-{seed}": _artifact(99, seed) for seed in range(3)}
    )

    with pytest.raises(StabilityTaskManifestError, match="prespecified manifest"):
        create_signed_stability_task_evidence(manifest, substituted, signer=_TestExternalAdapter())


def test_manifest_rejects_run_and_task_policy_substitution():
    reports = _reports()
    manifest = _manifest(reports)
    different_run_policy = copy.deepcopy(reports)
    different_run_policy["public-task-07"] = audit_artifact_stability_cohort(
        {f"seed-{seed}": _artifact(7, seed) for seed in range(3)},
        policy=StabilityCohortPolicy(min_pairwise_pearson=0.90),
    )
    with pytest.raises(StabilityTaskCohortError, match="different run-cohort policies"):
        create_signed_stability_task_evidence(
            manifest,
            different_run_policy,
            signer=_TestExternalAdapter(),
        )

    policy_manifest = _manifest(
        reports,
        task_cohort_policy=StabilityTaskCohortPolicy(confidence_level=0.90),
    )
    bundle = create_signed_stability_task_evidence(
        policy_manifest,
        reports,
        signer=_TestExternalAdapter(),
    )
    assert bundle["cohort_report"]["policy"]["confidence_level"] == 0.90


def test_signing_rejects_failed_cohort_and_sanitizes_provider_error():
    reports = _reports()
    manifest = _manifest(reports)
    rejected_policy = StabilityTaskCohortPolicy(min_accepted_task_rate_lower_bound=0.99)
    rejected_manifest = _manifest(reports, task_cohort_policy=rejected_policy)

    with pytest.raises(StabilityTaskManifestError, match="only accepted"):
        create_signed_stability_task_evidence(
            rejected_manifest,
            reports,
            signer=_TestExternalAdapter(),
        )

    with pytest.raises(StabilityTaskManifestError, match="external signing failed") as error:
        create_signed_stability_task_evidence(manifest, reports, signer=_FailingSigner())
    assert "provider diagnostic" not in str(error.value)


def test_verification_rejects_tampering_before_calling_external_verifier():
    reports = _reports()
    adapter = _TestExternalAdapter()
    bundle = create_signed_stability_task_evidence(_manifest(reports), reports, signer=adapter)

    tampered = copy.deepcopy(bundle)
    tampered["manifest"]["model_revision"] = "7" * 40
    with pytest.raises(StabilityTaskManifestError, match="manifest digest"):
        verify_signed_stability_task_evidence(tampered, {adapter.key_id: adapter})

    tampered = copy.deepcopy(bundle)
    tampered["cohort_report"]["accepted"] = False
    with pytest.raises(StabilityTaskManifestError, match="evidence digest"):
        verify_signed_stability_task_evidence(tampered, {adapter.key_id: adapter})


def test_verification_rejects_unknown_key_revision_and_bad_signature():
    reports = _reports()
    signer = _TestExternalAdapter()
    bundle = create_signed_stability_task_evidence(_manifest(reports), reports, signer=signer)

    with pytest.raises(StabilityTaskManifestError, match="unknown"):
        verify_signed_stability_task_evidence(bundle, {})

    wrong_revision = _TestExternalAdapter()
    wrong_revision.key_revision_sha256 = "8" * 64
    with pytest.raises(StabilityTaskManifestError, match="revision"):
        verify_signed_stability_task_evidence(bundle, {signer.key_id: wrong_revision})

    wrong_key = _TestExternalAdapter(secret=b"different-test-only-key")
    with pytest.raises(StabilityTaskManifestError, match="signature is invalid"):
        verify_signed_stability_task_evidence(bundle, {signer.key_id: wrong_key})


@pytest.mark.parametrize(
    "override,match",
    [
        ({"dataset_uri": "http://example.org/dataset"}, "HTTPS URI"),
        ({"dataset_uri": "https://user@example.org/dataset"}, "without credentials"),
        ({"model_revision": "main"}, "hexadecimal revision"),
        ({"selection_protocol_sha256": "not-a-digest"}, "canonical SHA-256"),
    ],
)
def test_manifest_requires_immutable_public_sources(override, match):
    with pytest.raises(StabilityTaskManifestError, match=match):
        _manifest(**override)


def test_manifest_validation_rejects_digest_and_schema_tampering():
    manifest = _manifest()
    manifest["dataset_records_sha256"] = "7" * 64
    with pytest.raises(StabilityTaskManifestError, match="manifest digest"):
        validate_stability_task_manifest(manifest)

    manifest = _manifest()
    manifest["unexpected"] = True
    with pytest.raises(StabilityTaskManifestError, match="exact schema"):
        validate_stability_task_manifest(manifest)
