from __future__ import annotations

import copy
import hashlib
import json

import pytest

from structxai.stability_lineage import create_manifest_publication_reference
from structxai.stability_reference_custody import (
    CUSTODIED_STABILITY_AUDIT_RECEIPT_SCHEMA,
    CUSTODY_KEY_STATUS_SNAPSHOT_SCHEMA,
    MAX_CUSTODY_ATTESTATION_LIFETIME_SECONDS,
    PUBLICATION_REFERENCE_CUSTODY_SCHEMA,
    StabilityReferenceCustodyError,
    create_custody_key_status_snapshot,
    create_publication_reference_custody_attestation,
    serialize_custodied_stability_audit_receipt,
    verify_custodied_stability_audit_bundle,
    verify_custody_key_status_snapshot,
    verify_publication_reference_custody_attestation,
    verify_publication_reference_custody_with_key_status,
)
from tests.test_stability_audit_bundle import PUBLICATION_URI, PUBLISHED_AT, _audit_fixture

ATTESTED_AT = PUBLISHED_AT + 60
EXPIRES_AT = ATTESTED_AT + 600
VERIFICATION_EPOCH = ATTESTED_AT + 30


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
    return {f"custodian-{index}": _CustodianAdapter(str(index)) for index in range(1, count + 1)}


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


def _attest(
    reference: dict[str, object],
    custodians: dict[str, _CustodianAdapter],
    *,
    required_approvals: int = 2,
) -> dict[str, object]:
    return create_publication_reference_custody_attestation(
        reference,
        custodians,
        required_approvals=required_approvals,
        attested_at_epoch=ATTESTED_AT,
        expires_at_epoch=EXPIRES_AT,
    )


def _key_status_snapshot(
    custodians: dict[str, _CustodianAdapter],
    *,
    statuses: dict[str, str] | None = None,
    issued_at_epoch: int = ATTESTED_AT,
    expires_at_epoch: int = EXPIRES_AT,
) -> tuple[dict[str, object], dict[str, _CustodianAdapter]]:
    authorities = {
        adapter.key_id: adapter
        for adapter in (
            _CustodianAdapter("key-status-authority-1"),
            _CustodianAdapter("key-status-authority-2"),
            _CustodianAdapter("key-status-authority-3"),
        )
    }
    values = statuses or {}
    snapshot = create_custody_key_status_snapshot(
        {
            adapter.key_id: (
                adapter.key_revision_sha256,
                values.get(adapter.key_id, "active"),
            )
            for adapter in custodians.values()
        },
        authorities,
        required_authority_signatures=2,
        signing_authority_ids=(
            "reference-custody-key-status-authority-1",
            "reference-custody-key-status-authority-3",
        ),
        issued_at_epoch=issued_at_epoch,
        expires_at_epoch=expires_at_epoch,
    )
    return snapshot, authorities


def test_custodied_audit_is_deterministic_cross_bound_and_privacy_reduced(tmp_path) -> None:
    manifest, record, reference, lineage, lineage_adapter = _audit_fixture(tmp_path)
    custodians = _custodians()
    attestation = _attest(reference, custodians)

    first = verify_custodied_stability_audit_bundle(
        reference,
        attestation,
        record,
        retrieved_uri=PUBLICATION_URI,
        lineage_bundle=lineage,
        custody_verifiers=_verifiers(custodians),
        lineage_verifiers={lineage_adapter.key_id: lineage_adapter},
        verification_epoch=VERIFICATION_EPOCH,
    )
    second = verify_custodied_stability_audit_bundle(
        reference,
        attestation,
        record,
        retrieved_uri=PUBLICATION_URI,
        lineage_bundle=lineage,
        custody_verifiers=_verifiers(custodians),
        lineage_verifiers={lineage_adapter.key_id: lineage_adapter},
        verification_epoch=VERIFICATION_EPOCH,
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
    attestation = _attest(reference, custodians)

    with_one_unavailable = verify_publication_reference_custody_attestation(
        reference,
        attestation,
        {
            custodians["custodian-1"].key_id: custodians["custodian-1"],
            custodians["custodian-2"].key_id: custodians["custodian-2"],
        },
        verification_epoch=VERIFICATION_EPOCH,
    )
    tampered = copy.deepcopy(attestation)
    signature = tampered["approvals"][2]["signature"]
    tampered["approvals"][2]["signature"] = ("A" if signature[0] != "A" else "B") + signature[1:]
    tampered["attestation_sha256"] = _digest(
        {key: value for key, value in tampered.items() if key != "attestation_sha256"}
    )
    with_one_invalid = verify_publication_reference_custody_attestation(
        reference,
        tampered,
        _verifiers(custodians),
        verification_epoch=VERIFICATION_EPOCH,
    )

    assert with_one_unavailable["valid_approvals"] == 2
    assert with_one_invalid["valid_approvals"] == 2
    assert with_one_unavailable["attested_at_epoch"] == ATTESTED_AT
    assert with_one_unavailable["expires_at_epoch"] == EXPIRES_AT
    assert with_one_unavailable["verified_at_epoch"] == VERIFICATION_EPOCH


def test_verification_rejects_expired_and_future_attestations(tmp_path) -> None:
    _manifest, _record, reference, _lineage, _lineage_adapter = _audit_fixture(tmp_path)
    custodians = _custodians()
    attestation = _attest(reference, custodians)

    with pytest.raises(StabilityReferenceCustodyError, match="has expired"):
        verify_publication_reference_custody_attestation(
            reference,
            attestation,
            _verifiers(custodians),
            verification_epoch=EXPIRES_AT + 1,
        )
    with pytest.raises(StabilityReferenceCustodyError, match="from the future"):
        verify_publication_reference_custody_attestation(
            reference,
            attestation,
            _verifiers(custodians),
            verification_epoch=ATTESTED_AT - 301,
        )


def test_validity_window_is_bound_into_every_signature(tmp_path) -> None:
    _manifest, _record, reference, _lineage, _lineage_adapter = _audit_fixture(tmp_path)
    custodians = _custodians()
    attestation = _attest(reference, custodians)
    tampered = copy.deepcopy(attestation)
    tampered["expires_at_epoch"] = EXPIRES_AT + 1
    tampered["attestation_sha256"] = _digest(
        {key: value for key, value in tampered.items() if key != "attestation_sha256"}
    )

    with pytest.raises(StabilityReferenceCustodyError, match="quorum is not satisfied"):
        verify_publication_reference_custody_attestation(
            reference,
            tampered,
            _verifiers(custodians),
            verification_epoch=VERIFICATION_EPOCH,
        )


@pytest.mark.parametrize(
    ("attested_at_epoch", "expires_at_epoch"),
    [
        (PUBLISHED_AT - 1, PUBLISHED_AT + 60),
        (ATTESTED_AT, ATTESTED_AT),
        (ATTESTED_AT, ATTESTED_AT + MAX_CUSTODY_ATTESTATION_LIFETIME_SECONDS + 1),
        (True, EXPIRES_AT),
    ],
)
def test_creation_rejects_invalid_freshness_windows(
    tmp_path,
    attested_at_epoch,
    expires_at_epoch,
) -> None:
    _manifest, _record, reference, _lineage, _lineage_adapter = _audit_fixture(tmp_path)

    with pytest.raises(StabilityReferenceCustodyError, match="invalid"):
        create_publication_reference_custody_attestation(
            reference,
            _custodians(),
            attested_at_epoch=attested_at_epoch,
            expires_at_epoch=expires_at_epoch,
        )


def test_quorum_rejects_reference_substitution_and_too_few_valid_keys(tmp_path) -> None:
    _manifest, _record, reference, _lineage, _lineage_adapter = _audit_fixture(tmp_path)
    custodians = _custodians()
    attestation = _attest(reference, custodians)
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
            verification_epoch=VERIFICATION_EPOCH,
        )
    with pytest.raises(StabilityReferenceCustodyError, match="quorum is not satisfied"):
        verify_publication_reference_custody_attestation(
            reference,
            attestation,
            {custodians["custodian-1"].key_id: custodians["custodian-1"]},
            verification_epoch=VERIFICATION_EPOCH,
        )


def test_custodian_set_and_threshold_are_bound_into_each_signature(tmp_path) -> None:
    _manifest, _record, reference, _lineage, _lineage_adapter = _audit_fixture(tmp_path)
    custodians = _custodians()
    attestation = _attest(reference, custodians)
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
            verification_epoch=VERIFICATION_EPOCH,
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
            attested_at_epoch=ATTESTED_AT,
            expires_at_epoch=EXPIRES_AT,
        )
    with pytest.raises(StabilityReferenceCustodyError, match="external custody signing failed") as error:
        create_publication_reference_custody_attestation(
            reference,
            {
                "custodian-1": _CustodianAdapter("1"),
                "custodian-2": _FailingCustodian("2"),
            },
            attested_at_epoch=ATTESTED_AT,
            expires_at_epoch=EXPIRES_AT,
        )

    assert "private provider diagnostic" not in str(error.value)


def test_attestation_and_receipt_reject_schema_extension_and_tampering(tmp_path) -> None:
    _manifest, record, reference, lineage, lineage_adapter = _audit_fixture(tmp_path)
    custodians = _custodians(2)
    attestation = _attest(reference, custodians)
    assert attestation["schema_version"] == PUBLICATION_REFERENCE_CUSTODY_SCHEMA

    with pytest.raises(StabilityReferenceCustodyError, match="exact schema"):
        verify_publication_reference_custody_attestation(
            reference,
            {**attestation, "unreviewed": True},
            _verifiers(custodians),
            verification_epoch=VERIFICATION_EPOCH,
        )

    receipt = verify_custodied_stability_audit_bundle(
        reference,
        attestation,
        record,
        retrieved_uri=PUBLICATION_URI,
        lineage_bundle=lineage,
        custody_verifiers=_verifiers(custodians),
        lineage_verifiers={lineage_adapter.key_id: lineage_adapter},
        verification_epoch=VERIFICATION_EPOCH,
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


def test_key_status_snapshot_filters_revoked_revisions_before_quorum(tmp_path) -> None:
    _manifest, _record, reference, _lineage, _lineage_adapter = _audit_fixture(tmp_path)
    custodians = _custodians()
    attestation = _attest(reference, custodians)
    revoked_key_id = custodians["custodian-3"].key_id
    snapshot, authorities = _key_status_snapshot(
        custodians,
        statuses={revoked_key_id: "revoked"},
    )

    verification = verify_publication_reference_custody_with_key_status(
        reference,
        attestation,
        _verifiers(custodians),
        snapshot,
        authorities,
        verification_epoch=VERIFICATION_EPOCH,
    )

    assert snapshot["schema_version"] == CUSTODY_KEY_STATUS_SNAPSHOT_SCHEMA
    assert verification["valid_approvals"] == 2
    assert verification["active_key_count"] == 2
    assert verification["revoked_key_count"] == 1
    assert len(str(verification["key_status_verification_sha256"])) == 64
    status_verification = verify_custody_key_status_snapshot(
        snapshot,
        {adapter.key_id: adapter.key_revision_sha256 for adapter in custodians.values()},
        authorities,
        verification_epoch=VERIFICATION_EPOCH,
    )
    assert status_verification["required_authority_signatures"] == 2
    assert status_verification["valid_authority_signatures"] == 2


def test_revoked_quorum_cannot_verify_even_with_valid_custody_signatures(tmp_path) -> None:
    _manifest, _record, reference, _lineage, _lineage_adapter = _audit_fixture(tmp_path)
    custodians = _custodians()
    attestation = _attest(reference, custodians)
    snapshot, authorities = _key_status_snapshot(
        custodians,
        statuses={
            custodians["custodian-2"].key_id: "revoked",
            custodians["custodian-3"].key_id: "revoked",
        },
    )

    with pytest.raises(StabilityReferenceCustodyError, match="quorum is not satisfied"):
        verify_publication_reference_custody_with_key_status(
            reference,
            attestation,
            _verifiers(custodians),
            snapshot,
            authorities,
            verification_epoch=VERIFICATION_EPOCH,
        )


def test_key_status_snapshot_rejects_tampering_coverage_and_authority_substitution(tmp_path) -> None:
    _manifest, _record, _reference, _lineage, _lineage_adapter = _audit_fixture(tmp_path)
    custodians = _custodians()
    snapshot, authorities = _key_status_snapshot(custodians)
    tampered = copy.deepcopy(snapshot)
    tampered["statuses"][0]["status"] = "revoked"
    missing = copy.deepcopy(snapshot)
    missing["statuses"].pop()
    missing["snapshot_sha256"] = _digest(
        {
            key: value
            for key, value in missing.items()
            if key not in {"snapshot_sha256", "authority_signatures"}
        }
    )
    wrong_authorities = {
        adapter.key_id: adapter
        for adapter in (
            _CustodianAdapter("wrong-key-status-authority-1"),
            _CustodianAdapter("wrong-key-status-authority-2"),
        )
    }
    expected_keys = {adapter.key_id: adapter.key_revision_sha256 for adapter in custodians.values()}

    with pytest.raises(StabilityReferenceCustodyError, match="digest does not match"):
        verify_custody_key_status_snapshot(
            tampered,
            expected_keys,
            authorities,
            verification_epoch=VERIFICATION_EPOCH,
        )
    with pytest.raises(StabilityReferenceCustodyError, match="does not cover"):
        verify_custody_key_status_snapshot(
            missing,
            expected_keys,
            authorities,
            verification_epoch=VERIFICATION_EPOCH,
        )
    with pytest.raises(StabilityReferenceCustodyError, match="policy mismatch"):
        verify_custody_key_status_snapshot(
            snapshot,
            expected_keys,
            wrong_authorities,
            verification_epoch=VERIFICATION_EPOCH,
        )


def test_key_status_snapshot_enforces_authority_quorum_and_exact_policy(tmp_path) -> None:
    _manifest, _record, _reference, _lineage, _lineage_adapter = _audit_fixture(tmp_path)
    custodians = _custodians()
    snapshot, authorities = _key_status_snapshot(custodians)
    expected_keys = {adapter.key_id: adapter.key_revision_sha256 for adapter in custodians.values()}
    missing_signature = copy.deepcopy(snapshot)
    missing_signature["authority_signatures"].pop()
    tampered_signature = copy.deepcopy(snapshot)
    signature = str(tampered_signature["authority_signatures"][0]["signature"])
    tampered_signature["authority_signatures"][0]["signature"] = (
        "A" if signature[0] != "A" else "B"
    ) + signature[1:]

    with pytest.raises(StabilityReferenceCustodyError, match="authority signatures are invalid"):
        verify_custody_key_status_snapshot(
            missing_signature,
            expected_keys,
            authorities,
            verification_epoch=VERIFICATION_EPOCH,
        )
    with pytest.raises(StabilityReferenceCustodyError, match="authority signature is invalid"):
        verify_custody_key_status_snapshot(
            tampered_signature,
            expected_keys,
            authorities,
            verification_epoch=VERIFICATION_EPOCH,
        )
    with pytest.raises(StabilityReferenceCustodyError, match="threshold mismatch"):
        verify_custody_key_status_snapshot(
            snapshot,
            expected_keys,
            authorities,
            required_authority_signatures=3,
            verification_epoch=VERIFICATION_EPOCH,
        )


def test_key_status_authority_policy_rejects_aliases_and_insufficient_signers(tmp_path) -> None:
    _manifest, _record, _reference, _lineage, _lineage_adapter = _audit_fixture(tmp_path)
    custodians = _custodians()
    first = _CustodianAdapter("key-status-authority-shared")
    alias = _CustodianAdapter("key-status-authority-shared")
    alias.key_id = "reference-custody-key-status-authority-alias"
    authorities = {first.key_id: first, alias.key_id: alias}
    statuses = {adapter.key_id: (adapter.key_revision_sha256, "active") for adapter in custodians.values()}

    with pytest.raises(StabilityReferenceCustodyError, match="revisions must be distinct"):
        create_custody_key_status_snapshot(
            statuses,
            authorities,
            issued_at_epoch=ATTESTED_AT,
            expires_at_epoch=EXPIRES_AT,
        )
    distinct_authorities = {
        adapter.key_id: adapter
        for adapter in (
            _CustodianAdapter("key-status-authority-1"),
            _CustodianAdapter("key-status-authority-2"),
            _CustodianAdapter("key-status-authority-3"),
        )
    }
    with pytest.raises(StabilityReferenceCustodyError, match="signing authority set is invalid"):
        create_custody_key_status_snapshot(
            statuses,
            distinct_authorities,
            required_authority_signatures=2,
            signing_authority_ids=("reference-custody-key-status-authority-1",),
            issued_at_epoch=ATTESTED_AT,
            expires_at_epoch=EXPIRES_AT,
        )


@pytest.mark.parametrize(
    ("issued_at_epoch", "expires_at_epoch", "verification_epoch", "message"),
    [
        (ATTESTED_AT, EXPIRES_AT, EXPIRES_AT + 1, "has expired"),
        (ATTESTED_AT, EXPIRES_AT, ATTESTED_AT - 301, "from the future"),
        (
            ATTESTED_AT,
            ATTESTED_AT + MAX_CUSTODY_ATTESTATION_LIFETIME_SECONDS + 1,
            VERIFICATION_EPOCH,
            "validity window",
        ),
    ],
)
def test_key_status_snapshot_enforces_freshness(
    tmp_path,
    issued_at_epoch,
    expires_at_epoch,
    verification_epoch,
    message,
) -> None:
    _manifest, _record, _reference, _lineage, _lineage_adapter = _audit_fixture(tmp_path)
    custodians = _custodians()
    expected_keys = {adapter.key_id: adapter.key_revision_sha256 for adapter in custodians.values()}
    if expires_at_epoch > issued_at_epoch + MAX_CUSTODY_ATTESTATION_LIFETIME_SECONDS:
        with pytest.raises(StabilityReferenceCustodyError, match=message):
            _key_status_snapshot(
                custodians,
                issued_at_epoch=issued_at_epoch,
                expires_at_epoch=expires_at_epoch,
            )
        return
    snapshot, authorities = _key_status_snapshot(
        custodians,
        issued_at_epoch=issued_at_epoch,
        expires_at_epoch=expires_at_epoch,
    )
    with pytest.raises(StabilityReferenceCustodyError, match=message):
        verify_custody_key_status_snapshot(
            snapshot,
            expected_keys,
            authorities,
            verification_epoch=verification_epoch,
        )


def test_key_status_authority_must_be_independent_from_custodian_keys(tmp_path) -> None:
    _manifest, _record, _reference, _lineage, _lineage_adapter = _audit_fixture(tmp_path)
    custodians = _custodians()
    custodian_authority = custodians["custodian-1"]
    independent_authority = _CustodianAdapter("independent-key-status-authority")
    authorities = {
        custodian_authority.key_id: custodian_authority,
        independent_authority.key_id: independent_authority,
    }

    with pytest.raises(StabilityReferenceCustodyError, match="must be independent"):
        create_custody_key_status_snapshot(
            {adapter.key_id: (adapter.key_revision_sha256, "active") for adapter in custodians.values()},
            authorities,
            issued_at_epoch=ATTESTED_AT,
            expires_at_epoch=EXPIRES_AT,
        )


def test_custodied_audit_can_cross_bind_key_status_verification(tmp_path) -> None:
    _manifest, record, reference, lineage, lineage_adapter = _audit_fixture(tmp_path)
    custodians = _custodians()
    attestation = _attest(reference, custodians)
    snapshot, authorities = _key_status_snapshot(custodians)

    receipt = verify_custodied_stability_audit_bundle(
        reference,
        attestation,
        record,
        retrieved_uri=PUBLICATION_URI,
        lineage_bundle=lineage,
        custody_verifiers=_verifiers(custodians),
        lineage_verifiers={lineage_adapter.key_id: lineage_adapter},
        verification_epoch=VERIFICATION_EPOCH,
        key_status_snapshot=snapshot,
        key_status_authority_verifiers=authorities,
    )

    assert receipt["verified"] is True
    assert receipt["reference_valid_approvals"] == 3
    assert len(str(receipt["reference_custody_verification_sha256"])) == 64
