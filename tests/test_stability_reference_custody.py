from __future__ import annotations

import copy
import hashlib
import json

import pytest

from structxai.stability_lineage import create_manifest_publication_reference
from structxai.stability_reference_custody import (
    CUSTODIED_STABILITY_AUDIT_RECEIPT_SCHEMA,
    PUBLICATION_REFERENCE_CUSTODY_SCHEMA,
    StabilityReferenceCustodyError,
    create_publication_reference_custody_attestation,
    serialize_custodied_stability_audit_receipt,
    verify_custodied_stability_audit_bundle,
    verify_publication_reference_custody_attestation,
)
from tests.test_stability_audit_bundle import PUBLICATION_URI, _audit_fixture


class _CustodianAdapter:
    algorithm = "ED25519"

    def __init__(self, custodian: str) -> None:
        self.key_id = f"reference-custody-{custodian}"
        self.key_revision_sha256 = hashlib.sha256(f"public-key-{custodian}".encode()).hexdigest()
        self._secret = f"test-only-secret-{custodian}".encode()

    def sign(self, message: bytes) -> bytes:
        return hashlib.sha512(self._secret + message).digest()

    def verify(self, message: bytes, signature: bytes) -> bool:
        return signature == hashlib.sha512(self._secret + message).digest()


class _FailingCustodian(_CustodianAdapter):
    def sign(self, message: bytes) -> bytes:
        raise RuntimeError("private provider diagnostic")


def _custodians(count: int = 3) -> dict[str, _CustodianAdapter]:
    return {
        f"custodian-{index}": _CustodianAdapter(str(index))
        for index in range(1, count + 1)
    }


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _verifiers(custodians: dict[str, _CustodianAdapter]) -> dict[str, _CustodianAdapter]:
    return {adapter.key_id: adapter for adapter in custodians.values()}


def test_custodied_audit_is_deterministic_cross_bound_and_privacy_reduced(tmp_path) -> None:
    manifest, record, reference, lineage, lineage_adapter = _audit_fixture(tmp_path)
    custodians = _custodians()
    attestation = create_publication_reference_custody_attestation(
        reference,
        custodians,
        required_approvals=2,
    )

    first = verify_custodied_stability_audit_bundle(
        reference,
        attestation,
        record,
        retrieved_uri=PUBLICATION_URI,
        lineage_bundle=lineage,
        custody_verifiers=_verifiers(custodians),
        lineage_verifiers={lineage_adapter.key_id: lineage_adapter},
    )
    second = verify_custodied_stability_audit_bundle(
        reference,
        attestation,
        record,
        retrieved_uri=PUBLICATION_URI,
        lineage_bundle=lineage,
        custody_verifiers=_verifiers(custodians),
        lineage_verifiers={lineage_adapter.key_id: lineage_adapter},
    )
    payload = serialize_custodied_stability_audit_receipt(first)

    assert first == second
    assert first["schema_version"] == CUSTODIED_STABILITY_AUDIT_RECEIPT_SCHEMA
    assert first["verified"] is True
    assert first["manifest_sha256"] == manifest["manifest_sha256"]
    assert first["reference_required_approvals"] == 2
    assert first["reference_valid_approvals"] == 3
    assert first["publication_reference_sha256"] == attestation["publication_reference_sha256"]
    assert len(str(first["receipt_sha256"])) == 64
    assert PUBLICATION_URI.encode() not in payload
    assert b"custodian-1" not in payload
    assert custodians["custodian-1"].key_id.encode() not in payload
    assert b"signature" not in payload


def test_quorum_accepts_one_unavailable_or_invalid_custodian(tmp_path) -> None:
    _manifest, _record, reference, _lineage, _lineage_adapter = _audit_fixture(tmp_path)
    custodians = _custodians()
    attestation = create_publication_reference_custody_attestation(
        reference,
        custodians,
        required_approvals=2,
    )

    with_one_unavailable = verify_publication_reference_custody_attestation(
        reference,
        attestation,
        {
            custodians["custodian-1"].key_id: custodians["custodian-1"],
            custodians["custodian-2"].key_id: custodians["custodian-2"],
        },
    )
    tampered = copy.deepcopy(attestation)
    signature = tampered["approvals"][2]["signature"]
    tampered["approvals"][2]["signature"] = (
        ("A" if signature[0] != "A" else "B") + signature[1:]
    )
    tampered["attestation_sha256"] = _digest(
        {key: value for key, value in tampered.items() if key != "attestation_sha256"}
    )
    with_one_invalid = verify_publication_reference_custody_attestation(
        reference,
        tampered,
        _verifiers(custodians),
    )

    assert with_one_unavailable["valid_approvals"] == 2
    assert with_one_invalid["valid_approvals"] == 2


def test_quorum_rejects_reference_substitution_and_too_few_valid_keys(tmp_path) -> None:
    _manifest, _record, reference, _lineage, _lineage_adapter = _audit_fixture(tmp_path)
    custodians = _custodians()
    attestation = create_publication_reference_custody_attestation(
        reference,
        custodians,
        required_approvals=2,
    )
    substituted = create_manifest_publication_reference(
        publication_uri="https://mirror.example/struct-xai/public-cohort-v1.json",
        publication_record_sha256=str(reference["publication_record_sha256"]),
        manifest_sha256=str(reference["manifest_sha256"]),
        published_at_epoch=int(reference["published_at_epoch"]),
    )

    with pytest.raises(StabilityReferenceCustodyError, match="different publication reference"):
        verify_publication_reference_custody_attestation(
            substituted,
            attestation,
            _verifiers(custodians),
        )
    with pytest.raises(StabilityReferenceCustodyError, match="quorum is not satisfied"):
        verify_publication_reference_custody_attestation(
            reference,
            attestation,
            {custodians["custodian-1"].key_id: custodians["custodian-1"]},
        )


def test_custodian_set_and_threshold_are_bound_into_each_signature(tmp_path) -> None:
    _manifest, _record, reference, _lineage, _lineage_adapter = _audit_fixture(tmp_path)
    custodians = _custodians()
    attestation = create_publication_reference_custody_attestation(
        reference,
        custodians,
        required_approvals=2,
    )
    tampered = copy.deepcopy(attestation)
    tampered["required_approvals"] = 3
    tampered["attestation_sha256"] = _digest(
        {key: value for key, value in tampered.items() if key != "attestation_sha256"}
    )

    with pytest.raises(StabilityReferenceCustodyError, match="quorum is not satisfied"):
        verify_publication_reference_custody_attestation(
            reference,
            tampered,
            _verifiers(custodians),
        )


def test_creation_requires_independent_key_revisions_and_sanitizes_provider_errors(tmp_path) -> None:
    _manifest, _record, reference, _lineage, _lineage_adapter = _audit_fixture(tmp_path)
    first = _CustodianAdapter("shared")
    second = _CustodianAdapter("shared")
    second.key_id = "reference-custody-alias"

    with pytest.raises(StabilityReferenceCustodyError, match="key revisions must be distinct"):
        create_publication_reference_custody_attestation(
            reference,
            {"custodian-1": first, "custodian-2": second},
        )
    with pytest.raises(StabilityReferenceCustodyError, match="external custody signing failed") as error:
        create_publication_reference_custody_attestation(
            reference,
            {
                "custodian-1": _CustodianAdapter("1"),
                "custodian-2": _FailingCustodian("2"),
            },
        )

    assert "private provider diagnostic" not in str(error.value)


def test_attestation_and_receipt_reject_schema_extension_and_tampering(tmp_path) -> None:
    _manifest, record, reference, lineage, lineage_adapter = _audit_fixture(tmp_path)
    custodians = _custodians(2)
    attestation = create_publication_reference_custody_attestation(reference, custodians)
    assert attestation["schema_version"] == PUBLICATION_REFERENCE_CUSTODY_SCHEMA

    with pytest.raises(StabilityReferenceCustodyError, match="exact schema"):
        verify_publication_reference_custody_attestation(
            reference,
            {**attestation, "unreviewed": True},
            _verifiers(custodians),
        )

    receipt = verify_custodied_stability_audit_bundle(
        reference,
        attestation,
        record,
        retrieved_uri=PUBLICATION_URI,
        lineage_bundle=lineage,
        custody_verifiers=_verifiers(custodians),
        lineage_verifiers={lineage_adapter.key_id: lineage_adapter},
    )
    with pytest.raises(StabilityReferenceCustodyError, match="digest does not match"):
        serialize_custodied_stability_audit_receipt(
            {
                **receipt,
                "reference_valid_approvals": int(receipt["reference_valid_approvals"]) + 1,
            }
        )
    with pytest.raises(StabilityReferenceCustodyError, match="exact schema"):
        serialize_custodied_stability_audit_receipt({**receipt, "unreviewed": True})
