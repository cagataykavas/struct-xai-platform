from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from typing import Protocol
from urllib.parse import urlsplit

from structxai.stability_uncertainty import (
    STABILITY_TASK_COHORT_SCHEMA,
    StabilityTaskCohortPolicy,
    audit_stability_task_cohort,
)

STABILITY_TASK_MANIFEST_SCHEMA = "struct-xai-stability-task-manifest/v1"
SIGNED_STABILITY_TASK_EVIDENCE_SCHEMA = "struct-xai-signed-stability-task-evidence/v1"
STABILITY_TASK_EVIDENCE_VERIFICATION_SCHEMA = "struct-xai-stability-task-evidence-verification/v1"
STABILITY_TASK_EVIDENCE_PURPOSE = "prespecified-public-task-stability-evidence"
EXTERNAL_SIGNATURE_ENCODING = "base64url-no-padding"
EXTERNAL_SIGNATURE_ALGORITHMS = frozenset(
    {
        "ECDSA-P256-SHA256",
        "ED25519",
        "RSASSA-PKCS1-V1_5-SHA256",
        "RSASSA-PSS-SHA256",
    }
)
MIN_EXTERNAL_SIGNATURE_BYTES = 32
MAX_EXTERNAL_SIGNATURE_BYTES = 8_192

_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_REVISION_RE = re.compile(r"[0-9a-f]{40,64}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_TASK_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_MANIFEST_FIELDS = {
    "schema_version",
    "cohort_id",
    "dataset_uri",
    "dataset_revision",
    "dataset_records_sha256",
    "selection_protocol_uri",
    "selection_protocol_sha256",
    "model_uri",
    "model_revision",
    "model_weights_sha256",
    "tokenizer_sha256",
    "run_cohort_policy_sha256",
    "task_cohort_policy",
    "task_cohort_policy_sha256",
    "tasks",
    "manifest_sha256",
}
_BUNDLE_FIELDS = {
    "schema_version",
    "purpose",
    "algorithm",
    "signature_encoding",
    "key_id",
    "key_revision_sha256",
    "manifest_sha256",
    "cohort_evidence_sha256",
    "task_count",
    "manifest",
    "cohort_report",
    "signature",
}
_SIGNED_CLAIM_FIELDS = _BUNDLE_FIELDS - {"manifest", "cohort_report", "signature"}


class ExternalEvidenceSigner(Protocol):
    """Structural interface implemented by the existing KMS/HSM adapters."""

    algorithm: str
    key_id: str
    key_revision_sha256: str

    def sign(self, message: bytes) -> bytes: ...


class ExternalEvidenceVerifier(Protocol):
    """Structural interface for one immutable external public-key revision."""

    algorithm: str
    key_id: str
    key_revision_sha256: str

    def verify(self, message: bytes, signature: bytes) -> bool: ...


class StabilityTaskManifestError(ValueError):
    """A prespecified cohort manifest or its signed evidence is invalid."""

    code = "STABILITY_TASK_MANIFEST_REJECTED"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise StabilityTaskManifestError("document is not canonical JSON") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _validate_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise StabilityTaskManifestError(f"{name} must be a canonical SHA-256")
    return value


def _validate_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise StabilityTaskManifestError(f"{name} must be a canonical identifier")
    return value


def _validate_uri(value: object, name: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value.encode("utf-8")) <= 2_048:
        raise StabilityTaskManifestError(f"{name} must be a bounded HTTPS URI")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise StabilityTaskManifestError(f"{name} must be a public HTTPS URI without credentials or fragment")
    return value


def _validate_revision(value: object, name: str) -> str:
    if not isinstance(value, str) or _REVISION_RE.fullmatch(value) is None:
        raise StabilityTaskManifestError(f"{name} must be a 40- to 64-character hexadecimal revision")
    return value


def _validate_policy(value: object, digest: object) -> StabilityTaskCohortPolicy:
    if not isinstance(value, dict):
        raise StabilityTaskManifestError("task_cohort_policy must be a dictionary")
    if _digest(value) != _validate_digest(digest, "task_cohort_policy_sha256"):
        raise StabilityTaskManifestError("task cohort policy digest does not match its content")
    try:
        return StabilityTaskCohortPolicy(**value)
    except (TypeError, ValueError) as exc:
        raise StabilityTaskManifestError("task cohort policy is invalid") from exc


def _validated_tasks(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list) or not 8 <= len(value) <= 256:
        raise StabilityTaskManifestError("tasks must contain from 8 through 256 commitments")
    normalized: list[dict[str, str]] = []
    for task in value:
        if not isinstance(task, dict) or set(task) != {"task_id", "experiment_sha256"}:
            raise StabilityTaskManifestError("each task commitment must use the exact schema")
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or _TASK_ID_RE.fullmatch(task_id) is None:
            raise StabilityTaskManifestError("task_id must be a canonical identifier")
        normalized.append(
            {
                "task_id": task_id,
                "experiment_sha256": _validate_digest(
                    task.get("experiment_sha256"),
                    "experiment_sha256",
                ),
            }
        )
    if normalized != sorted(normalized, key=lambda item: item["task_id"]):
        raise StabilityTaskManifestError("task commitments must be sorted by task_id")
    if len({item["task_id"] for item in normalized}) != len(normalized):
        raise StabilityTaskManifestError("task IDs must be unique")
    if len({item["experiment_sha256"] for item in normalized}) != len(normalized):
        raise StabilityTaskManifestError("experiment commitments must be unique")
    return normalized


def validate_stability_task_manifest(manifest: object) -> dict[str, object]:
    """Validate and normalize one privacy-reduced, prespecified task manifest."""

    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_FIELDS:
        raise StabilityTaskManifestError("manifest must use the exact schema")
    if manifest.get("schema_version") != STABILITY_TASK_MANIFEST_SCHEMA:
        raise StabilityTaskManifestError("manifest schema version is unsupported")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if _digest(unsigned) != _validate_digest(manifest.get("manifest_sha256"), "manifest_sha256"):
        raise StabilityTaskManifestError("manifest digest does not match its content")
    normalized = {
        **manifest,
        "cohort_id": _validate_identifier(manifest.get("cohort_id"), "cohort_id"),
        "dataset_uri": _validate_uri(manifest.get("dataset_uri"), "dataset_uri"),
        "dataset_revision": _validate_revision(manifest.get("dataset_revision"), "dataset_revision"),
        "dataset_records_sha256": _validate_digest(
            manifest.get("dataset_records_sha256"), "dataset_records_sha256"
        ),
        "selection_protocol_uri": _validate_uri(
            manifest.get("selection_protocol_uri"), "selection_protocol_uri"
        ),
        "selection_protocol_sha256": _validate_digest(
            manifest.get("selection_protocol_sha256"), "selection_protocol_sha256"
        ),
        "model_uri": _validate_uri(manifest.get("model_uri"), "model_uri"),
        "model_revision": _validate_revision(manifest.get("model_revision"), "model_revision"),
        "model_weights_sha256": _validate_digest(
            manifest.get("model_weights_sha256"), "model_weights_sha256"
        ),
        "tokenizer_sha256": _validate_digest(manifest.get("tokenizer_sha256"), "tokenizer_sha256"),
        "run_cohort_policy_sha256": _validate_digest(
            manifest.get("run_cohort_policy_sha256"), "run_cohort_policy_sha256"
        ),
        "tasks": _validated_tasks(manifest.get("tasks")),
    }
    _validate_policy(
        manifest.get("task_cohort_policy"),
        manifest.get("task_cohort_policy_sha256"),
    )
    return normalized


def create_stability_task_manifest(
    task_commitments: Mapping[str, str],
    *,
    cohort_id: str,
    dataset_uri: str,
    dataset_revision: str,
    dataset_records_sha256: str,
    selection_protocol_uri: str,
    selection_protocol_sha256: str,
    model_uri: str,
    model_revision: str,
    model_weights_sha256: str,
    tokenizer_sha256: str,
    run_cohort_policy_sha256: str,
    task_cohort_policy: StabilityTaskCohortPolicy | None = None,
) -> dict[str, object]:
    """Create the task-selection commitment that must exist before evaluation."""

    if not isinstance(task_commitments, Mapping):
        raise TypeError("task_commitments must be a mapping")
    policy = task_cohort_policy or StabilityTaskCohortPolicy()
    if not isinstance(policy, StabilityTaskCohortPolicy):
        raise TypeError("task_cohort_policy must be a StabilityTaskCohortPolicy")
    policy_document = policy.to_audit_dict()
    manifest: dict[str, object] = {
        "schema_version": STABILITY_TASK_MANIFEST_SCHEMA,
        "cohort_id": cohort_id,
        "dataset_uri": dataset_uri,
        "dataset_revision": dataset_revision,
        "dataset_records_sha256": dataset_records_sha256,
        "selection_protocol_uri": selection_protocol_uri,
        "selection_protocol_sha256": selection_protocol_sha256,
        "model_uri": model_uri,
        "model_revision": model_revision,
        "model_weights_sha256": model_weights_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "run_cohort_policy_sha256": run_cohort_policy_sha256,
        "task_cohort_policy": policy_document,
        "task_cohort_policy_sha256": _digest(policy_document),
        "tasks": [
            {"task_id": task_id, "experiment_sha256": experiment_sha256}
            for task_id, experiment_sha256 in sorted(task_commitments.items())
        ],
    }
    manifest["manifest_sha256"] = _digest(manifest)
    return validate_stability_task_manifest(manifest)


def _validate_cohort_report(report: object) -> dict[str, object]:
    if not isinstance(report, dict) or report.get("schema_version") != STABILITY_TASK_COHORT_SCHEMA:
        raise StabilityTaskManifestError("cohort report schema version is unsupported")
    evidence_sha256 = _validate_digest(report.get("evidence_sha256"), "cohort evidence_sha256")
    unsigned = {key: value for key, value in report.items() if key != "evidence_sha256"}
    if _digest(unsigned) != evidence_sha256:
        raise StabilityTaskManifestError("cohort evidence digest does not match its content")
    if report.get("accepted") is not True:
        raise StabilityTaskManifestError("only accepted task-cohort evidence can be signed")
    tasks = report.get("tasks")
    if not isinstance(tasks, list) or report.get("task_count") != len(tasks):
        raise StabilityTaskManifestError("cohort task count does not match its task list")
    return report


def _bind_manifest_to_report(manifest: dict[str, object], report: dict[str, object]) -> None:
    expected_tasks = manifest["tasks"]
    actual_tasks = report["tasks"]
    assert isinstance(expected_tasks, list) and isinstance(actual_tasks, list)
    actual_commitments = [
        {"task_id": item.get("task_id"), "experiment_sha256": item.get("experiment_sha256")}
        if isinstance(item, dict)
        else None
        for item in actual_tasks
    ]
    if not hmac.compare_digest(_digest(expected_tasks), _digest(actual_commitments)):
        raise StabilityTaskManifestError("cohort tasks do not match the prespecified manifest")
    for manifest_field, report_field in (
        ("run_cohort_policy_sha256", "run_cohort_policy_sha256"),
        ("task_cohort_policy_sha256", "policy_sha256"),
    ):
        if not hmac.compare_digest(str(manifest[manifest_field]), str(report.get(report_field))):
            raise StabilityTaskManifestError(f"cohort {report_field} does not match the manifest")


def _adapter_metadata(
    adapter: ExternalEvidenceSigner | ExternalEvidenceVerifier,
) -> tuple[str, str, str]:
    try:
        algorithm = _validate_identifier(adapter.algorithm, "signing algorithm")
        key_id = _validate_identifier(adapter.key_id, "key_id")
        key_revision = _validate_digest(adapter.key_revision_sha256, "key_revision_sha256")
    except StabilityTaskManifestError:
        raise
    except Exception as exc:
        raise StabilityTaskManifestError("external signature adapter metadata is invalid") from exc
    if algorithm not in EXTERNAL_SIGNATURE_ALGORITHMS:
        raise StabilityTaskManifestError("external signature algorithm is unsupported")
    return algorithm, key_id, key_revision


def _encode_signature(value: object) -> str:
    if (
        not isinstance(value, bytes)
        or not MIN_EXTERNAL_SIGNATURE_BYTES <= len(value) <= MAX_EXTERNAL_SIGNATURE_BYTES
    ):
        raise StabilityTaskManifestError("external signature is invalid")
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode_signature(value: object) -> bytes:
    if not isinstance(value, str) or not value or "=" in value:
        raise StabilityTaskManifestError("external signature is invalid")
    try:
        decoded = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
    except (binascii.Error, ValueError, UnicodeEncodeError) as exc:
        raise StabilityTaskManifestError("external signature is invalid") from exc
    if _encode_signature(decoded) != value:
        raise StabilityTaskManifestError("external signature is invalid")
    return decoded


def _signed_claims(bundle: Mapping[str, object]) -> dict[str, object]:
    return {key: bundle[key] for key in _SIGNED_CLAIM_FIELDS}


def create_signed_stability_task_evidence(
    manifest: dict[str, object],
    task_reports: Mapping[str, dict],
    *,
    signer: ExternalEvidenceSigner,
) -> dict[str, object]:
    """Recompute, bind, and externally sign accepted task-level stability evidence."""

    normalized_manifest = validate_stability_task_manifest(manifest)
    policy = _validate_policy(
        normalized_manifest["task_cohort_policy"],
        normalized_manifest["task_cohort_policy_sha256"],
    )
    report = _validate_cohort_report(audit_stability_task_cohort(task_reports, policy=policy))
    _bind_manifest_to_report(normalized_manifest, report)
    algorithm, key_id, key_revision = _adapter_metadata(signer)
    if not callable(getattr(signer, "sign", None)):
        raise StabilityTaskManifestError("external signer is invalid")
    bundle: dict[str, object] = {
        "schema_version": SIGNED_STABILITY_TASK_EVIDENCE_SCHEMA,
        "purpose": STABILITY_TASK_EVIDENCE_PURPOSE,
        "algorithm": algorithm,
        "signature_encoding": EXTERNAL_SIGNATURE_ENCODING,
        "key_id": key_id,
        "key_revision_sha256": key_revision,
        "manifest_sha256": normalized_manifest["manifest_sha256"],
        "cohort_evidence_sha256": report["evidence_sha256"],
        "task_count": report["task_count"],
        "manifest": normalized_manifest,
        "cohort_report": report,
    }
    try:
        signature = signer.sign(_canonical_json(_signed_claims(bundle)))
    except Exception as exc:
        raise StabilityTaskManifestError("external signing failed") from exc
    bundle["signature"] = _encode_signature(signature)
    return bundle


def verify_signed_stability_task_evidence(
    bundle: Mapping[str, object],
    verifiers: Mapping[str, ExternalEvidenceVerifier],
) -> dict[str, object]:
    """Verify the signature plus manifest, report, policy, and task-set bindings."""

    if not isinstance(bundle, Mapping) or set(bundle) != _BUNDLE_FIELDS:
        raise StabilityTaskManifestError("signed evidence bundle must use the exact schema")
    if (
        bundle.get("schema_version") != SIGNED_STABILITY_TASK_EVIDENCE_SCHEMA
        or bundle.get("purpose") != STABILITY_TASK_EVIDENCE_PURPOSE
        or bundle.get("signature_encoding") != EXTERNAL_SIGNATURE_ENCODING
    ):
        raise StabilityTaskManifestError("signed evidence bundle metadata is invalid")
    manifest = validate_stability_task_manifest(bundle.get("manifest"))
    report = _validate_cohort_report(bundle.get("cohort_report"))
    _bind_manifest_to_report(manifest, report)
    if (
        not hmac.compare_digest(str(bundle.get("manifest_sha256")), str(manifest["manifest_sha256"]))
        or not hmac.compare_digest(str(bundle.get("cohort_evidence_sha256")), str(report["evidence_sha256"]))
        or bundle.get("task_count") != report.get("task_count")
    ):
        raise StabilityTaskManifestError("signed digest claims do not match the retained evidence")
    if not isinstance(verifiers, Mapping):
        raise StabilityTaskManifestError("verifiers must be a mapping")
    verifier = verifiers.get(str(bundle.get("key_id")))
    if verifier is None:
        raise StabilityTaskManifestError("signature key is unknown")
    algorithm, key_id, key_revision = _adapter_metadata(verifier)
    if (
        not hmac.compare_digest(algorithm, str(bundle.get("algorithm")))
        or not hmac.compare_digest(key_id, str(bundle.get("key_id")))
        or not hmac.compare_digest(key_revision, str(bundle.get("key_revision_sha256")))
    ):
        raise StabilityTaskManifestError("signature key revision does not match the bundle")
    if not callable(getattr(verifier, "verify", None)):
        raise StabilityTaskManifestError("external verifier is invalid")
    message = _canonical_json(_signed_claims(bundle))
    signature = _decode_signature(bundle.get("signature"))
    try:
        verified = verifier.verify(message, signature)
    except Exception as exc:
        raise StabilityTaskManifestError("external verification failed") from exc
    if verified is not True:
        raise StabilityTaskManifestError("signature is invalid")
    return {
        "schema_version": STABILITY_TASK_EVIDENCE_VERIFICATION_SCHEMA,
        "verified": True,
        "cohort_id": manifest["cohort_id"],
        "task_count": report["task_count"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cohort_evidence_sha256": report["evidence_sha256"],
        "algorithm": algorithm,
        "key_id": key_id,
        "key_revision_sha256": key_revision,
        "signature_input_sha256": hashlib.sha256(message).hexdigest(),
    }
