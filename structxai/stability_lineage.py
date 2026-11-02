from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from dataclasses import asdict

from structxai.stability_manifest import (
    EXTERNAL_SIGNATURE_ALGORITHMS,
    EXTERNAL_SIGNATURE_ENCODING,
    MAX_EXTERNAL_SIGNATURE_BYTES,
    MIN_EXTERNAL_SIGNATURE_BYTES,
    ExternalEvidenceSigner,
    ExternalEvidenceVerifier,
    StabilityTaskManifestError,
    create_signed_stability_task_evidence,
    validate_stability_task_manifest,
    verify_signed_stability_task_evidence,
)
from structxai.stability_runtime import (
    STABILITY_RUNTIME_EVIDENCE_SCHEMA,
    StabilityRuntimeEvidence,
)

RUNTIME_BOUND_STABILITY_EVIDENCE_SCHEMA = "struct-xai-runtime-bound-stability-evidence/v1"
RUNTIME_BOUND_STABILITY_EVIDENCE_PURPOSE = "runtime-bound-prespecified-task-stability-evidence"
RUNTIME_BOUND_STABILITY_VERIFICATION_SCHEMA = "struct-xai-runtime-bound-stability-verification/v1"

_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_RUNTIME_FIELDS = {
    "schema_version",
    "cohort_id",
    "manifest_sha256",
    "task_count",
    "dataset_records_sha256",
    "selection_protocol_sha256",
    "model_weights_sha256",
    "tokenizer_sha256",
    "input_commitment_sha256",
    "report_commitments_sha256",
    "postflight_verified",
    "evidence_sha256",
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
    "runtime_evidence_sha256",
    "report_commitments_sha256",
    "signed_evidence_sha256",
    "task_count",
    "signed_evidence",
    "runtime_evidence",
    "signature",
}
_SIGNED_CLAIM_FIELDS = _BUNDLE_FIELDS - {"signed_evidence", "runtime_evidence", "signature"}
_INPUT_DIGEST_FIELDS = (
    "dataset_records_sha256",
    "selection_protocol_sha256",
    "model_weights_sha256",
    "tokenizer_sha256",
)


class StabilityLineageError(ValueError):
    """Stable failure from the runtime-to-signed-evidence lineage boundary."""

    code = "STABILITY_LINEAGE_REJECTED"

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
        raise StabilityLineageError("lineage evidence is not canonical JSON") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and _DIGEST_RE.fullmatch(value) is not None


def _same(left: object, right: object) -> bool:
    return isinstance(left, str) and isinstance(right, str) and hmac.compare_digest(left, right)


def _adapter_metadata(
    adapter: ExternalEvidenceSigner | ExternalEvidenceVerifier,
) -> tuple[str, str, str]:
    try:
        algorithm = adapter.algorithm
        key_id = adapter.key_id
        key_revision = adapter.key_revision_sha256
    except Exception as exc:
        raise StabilityLineageError("external signature adapter metadata is invalid") from exc
    if (
        not isinstance(algorithm, str)
        or algorithm not in EXTERNAL_SIGNATURE_ALGORITHMS
        or not isinstance(key_id, str)
        or _IDENTIFIER_RE.fullmatch(key_id) is None
        or not _is_digest(key_revision)
    ):
        raise StabilityLineageError("external signature adapter metadata is invalid")
    return algorithm, key_id, key_revision


def _encode_signature(value: object) -> str:
    if (
        not isinstance(value, bytes)
        or not MIN_EXTERNAL_SIGNATURE_BYTES <= len(value) <= MAX_EXTERNAL_SIGNATURE_BYTES
    ):
        raise StabilityLineageError("external signature is invalid")
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode_signature(value: object) -> bytes:
    if not isinstance(value, str) or not value or "=" in value:
        raise StabilityLineageError("external signature is invalid")
    try:
        decoded = base64.b64decode(
            value + "=" * (-len(value) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, ValueError, UnicodeEncodeError) as exc:
        raise StabilityLineageError("external signature is invalid") from exc
    if _encode_signature(decoded) != value:
        raise StabilityLineageError("external signature is invalid")
    return decoded


def _runtime_document(evidence: object) -> dict[str, object]:
    if isinstance(evidence, StabilityRuntimeEvidence):
        return asdict(evidence)
    if isinstance(evidence, Mapping):
        return dict(evidence)
    raise StabilityLineageError("runtime evidence is invalid")


def _validate_runtime_evidence(
    evidence: object,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    document = _runtime_document(evidence)
    if set(document) != _RUNTIME_FIELDS:
        raise StabilityLineageError("runtime evidence must use the exact schema")
    if (
        document.get("schema_version") != STABILITY_RUNTIME_EVIDENCE_SCHEMA
        or document.get("postflight_verified") is not True
        or document.get("cohort_id") != manifest.get("cohort_id")
        or document.get("task_count") != len(manifest["tasks"])
        or not _same(document.get("manifest_sha256"), manifest.get("manifest_sha256"))
    ):
        raise StabilityLineageError("runtime evidence does not match the manifest")
    if any(
        not _is_digest(document.get(field)) or not _same(document.get(field), manifest.get(field))
        for field in _INPUT_DIGEST_FIELDS
    ):
        raise StabilityLineageError("runtime input commitments do not match the manifest")
    for field in (
        "input_commitment_sha256",
        "report_commitments_sha256",
        "evidence_sha256",
    ):
        if not _is_digest(document.get(field)):
            raise StabilityLineageError("runtime evidence contains an invalid digest")
    expected_input = {
        "manifest_sha256": manifest["manifest_sha256"],
        "task_count": document["task_count"],
        **{field: document[field] for field in _INPUT_DIGEST_FIELDS},
    }
    if not _same(document["input_commitment_sha256"], _digest(expected_input)):
        raise StabilityLineageError("runtime input commitment is invalid")
    unsigned = {key: value for key, value in document.items() if key != "evidence_sha256"}
    if not _same(document["evidence_sha256"], _digest(unsigned)):
        raise StabilityLineageError("runtime evidence digest does not match its content")
    return document


def _report_commitments(
    reports: object,
    manifest: Mapping[str, object],
) -> tuple[dict[str, dict], str]:
    if not isinstance(reports, Mapping):
        raise StabilityLineageError("task reports are invalid")
    tasks = manifest["tasks"]
    assert isinstance(tasks, list)
    expected = {task["task_id"]: task["experiment_sha256"] for task in tasks}
    if set(reports) != set(expected):
        raise StabilityLineageError("task reports do not match the manifest")
    normalized: dict[str, dict] = {}
    commitments: list[dict[str, str]] = []
    for task_id in sorted(expected):
        report = reports[task_id]
        if not isinstance(report, dict) or not _same(report.get("experiment_sha256"), expected[task_id]):
            raise StabilityLineageError("task report does not match its commitment")
        canonical_report = _canonical_json(report)
        normalized_report = json.loads(canonical_report)
        if not isinstance(normalized_report, dict):
            raise StabilityLineageError("task reports are invalid")
        normalized[task_id] = normalized_report
        commitments.append(
            {
                "task_id": task_id,
                "experiment_sha256": expected[task_id],
                "report_sha256": hashlib.sha256(canonical_report).hexdigest(),
            }
        )
    return normalized, _digest(commitments)


def _signed_claims(bundle: Mapping[str, object]) -> dict[str, object]:
    return {key: bundle[key] for key in _SIGNED_CLAIM_FIELDS}


def create_runtime_bound_stability_evidence(
    manifest: dict[str, object],
    task_reports: Mapping[str, dict],
    runtime_evidence: StabilityRuntimeEvidence | Mapping[str, object],
    *,
    signer: ExternalEvidenceSigner,
) -> dict[str, object]:
    """Sign one lineage envelope over runtime evidence and accepted cohort evidence."""

    try:
        normalized_manifest = validate_stability_task_manifest(manifest)
    except StabilityTaskManifestError as exc:
        raise StabilityLineageError("stability manifest is invalid") from exc
    runtime_document = _validate_runtime_evidence(runtime_evidence, normalized_manifest)
    normalized_reports, report_commitments_sha256 = _report_commitments(
        task_reports,
        normalized_manifest,
    )
    if not _same(runtime_document["report_commitments_sha256"], report_commitments_sha256):
        raise StabilityLineageError("runtime evidence does not commit to the supplied reports")
    try:
        signed_evidence = create_signed_stability_task_evidence(
            normalized_manifest,
            normalized_reports,
            signer=signer,
        )
    except StabilityTaskManifestError as exc:
        raise StabilityLineageError("signed cohort evidence is invalid") from exc
    algorithm, key_id, key_revision = _adapter_metadata(signer)
    if (
        not _same(signed_evidence.get("algorithm"), algorithm)
        or not _same(signed_evidence.get("key_id"), key_id)
        or not _same(signed_evidence.get("key_revision_sha256"), key_revision)
    ):
        raise StabilityLineageError("signature adapter changed during lineage creation")
    bundle: dict[str, object] = {
        "schema_version": RUNTIME_BOUND_STABILITY_EVIDENCE_SCHEMA,
        "purpose": RUNTIME_BOUND_STABILITY_EVIDENCE_PURPOSE,
        "algorithm": algorithm,
        "signature_encoding": EXTERNAL_SIGNATURE_ENCODING,
        "key_id": key_id,
        "key_revision_sha256": key_revision,
        "manifest_sha256": normalized_manifest["manifest_sha256"],
        "cohort_evidence_sha256": signed_evidence["cohort_evidence_sha256"],
        "runtime_evidence_sha256": runtime_document["evidence_sha256"],
        "report_commitments_sha256": report_commitments_sha256,
        "signed_evidence_sha256": _digest(signed_evidence),
        "task_count": len(normalized_manifest["tasks"]),
        "signed_evidence": signed_evidence,
        "runtime_evidence": runtime_document,
    }
    try:
        signature = signer.sign(_canonical_json(_signed_claims(bundle)))
    except Exception as exc:
        raise StabilityLineageError("external lineage signing failed") from exc
    bundle["signature"] = _encode_signature(signature)
    return bundle


def verify_runtime_bound_stability_evidence(
    bundle: Mapping[str, object],
    verifiers: Mapping[str, ExternalEvidenceVerifier],
) -> dict[str, object]:
    """Verify both retained artifacts, their cross-bindings, and the lineage signature."""

    if not isinstance(bundle, Mapping) or set(bundle) != _BUNDLE_FIELDS:
        raise StabilityLineageError("runtime-bound evidence must use the exact schema")
    if (
        bundle.get("schema_version") != RUNTIME_BOUND_STABILITY_EVIDENCE_SCHEMA
        or bundle.get("purpose") != RUNTIME_BOUND_STABILITY_EVIDENCE_PURPOSE
        or bundle.get("signature_encoding") != EXTERNAL_SIGNATURE_ENCODING
    ):
        raise StabilityLineageError("runtime-bound evidence metadata is invalid")
    signed_evidence = bundle.get("signed_evidence")
    if not isinstance(signed_evidence, Mapping):
        raise StabilityLineageError("signed cohort evidence is invalid")
    try:
        signed_verification = verify_signed_stability_task_evidence(signed_evidence, verifiers)
        manifest = validate_stability_task_manifest(signed_evidence.get("manifest"))
    except StabilityTaskManifestError as exc:
        raise StabilityLineageError("signed cohort evidence is invalid") from exc
    runtime_document = _validate_runtime_evidence(bundle.get("runtime_evidence"), manifest)
    for value in (
        bundle.get("runtime_evidence_sha256"),
        bundle.get("report_commitments_sha256"),
        bundle.get("signed_evidence_sha256"),
    ):
        if not _is_digest(value):
            raise StabilityLineageError("runtime-bound evidence contains an invalid digest")
    if (
        not _same(bundle.get("manifest_sha256"), signed_verification["manifest_sha256"])
        or not _same(
            bundle.get("cohort_evidence_sha256"),
            signed_verification["cohort_evidence_sha256"],
        )
        or bundle.get("task_count") != signed_verification["task_count"]
        or not _same(bundle.get("runtime_evidence_sha256"), runtime_document["evidence_sha256"])
        or not _same(
            bundle.get("report_commitments_sha256"),
            runtime_document["report_commitments_sha256"],
        )
        or not _same(bundle.get("signed_evidence_sha256"), _digest(signed_evidence))
        or not _same(bundle.get("algorithm"), signed_verification["algorithm"])
        or not _same(bundle.get("key_id"), signed_verification["key_id"])
        or not _same(
            bundle.get("key_revision_sha256"),
            signed_verification["key_revision_sha256"],
        )
    ):
        raise StabilityLineageError("runtime-bound digest claims do not match retained evidence")
    if not isinstance(verifiers, Mapping):
        raise StabilityLineageError("verifiers must be a mapping")
    verifier = verifiers.get(str(bundle.get("key_id")))
    if verifier is None:
        raise StabilityLineageError("lineage signature key is unknown")
    algorithm, key_id, key_revision = _adapter_metadata(verifier)
    if (
        not _same(algorithm, bundle.get("algorithm"))
        or not _same(key_id, bundle.get("key_id"))
        or not _same(key_revision, bundle.get("key_revision_sha256"))
    ):
        raise StabilityLineageError("lineage signature key revision does not match the bundle")
    message = _canonical_json(_signed_claims(bundle))
    signature = _decode_signature(bundle.get("signature"))
    try:
        verified = verifier.verify(message, signature)
    except Exception as exc:
        raise StabilityLineageError("external lineage verification failed") from exc
    if verified is not True:
        raise StabilityLineageError("lineage signature is invalid")
    return {
        "schema_version": RUNTIME_BOUND_STABILITY_VERIFICATION_SCHEMA,
        "verified": True,
        "cohort_id": runtime_document["cohort_id"],
        "task_count": signed_verification["task_count"],
        "manifest_sha256": signed_verification["manifest_sha256"],
        "cohort_evidence_sha256": signed_verification["cohort_evidence_sha256"],
        "runtime_evidence_sha256": runtime_document["evidence_sha256"],
        "report_commitments_sha256": runtime_document["report_commitments_sha256"],
        "signed_evidence_sha256": bundle["signed_evidence_sha256"],
        "algorithm": algorithm,
        "key_id": key_id,
        "key_revision_sha256": key_revision,
        "signature_input_sha256": hashlib.sha256(message).hexdigest(),
    }
