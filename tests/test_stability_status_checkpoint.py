from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from structxai.stability_status_checkpoint import (
    MAX_SIGNED_STATUS_CHECKPOINT_BYTES,
    SIGNED_STATUS_CHECKPOINT_VERIFICATION_SCHEMA,
    SignedCustodyStatusCheckpointError,
    create_signed_custody_status_checkpoint,
    load_signed_custody_status_checkpoint,
    serialize_signed_custody_status_checkpoint,
    verify_signed_custody_status_checkpoint,
)
from tests.test_stability_reference_custody import (
    ATTESTED_AT,
    VERIFICATION_EPOCH,
    _CustodianAdapter,
    _custodians,
    _key_status_snapshot,
)
from tests.test_stability_status_transparency import LOG_ID, _append

SOURCE_REVISION = "a" * 40
SIGNED_AT = VERIFICATION_EPOCH + 1


class _UnavailableVerifier(_CustodianAdapter):
    def verify(self, message: bytes, signature: bytes) -> bool:
        raise RuntimeError("private provider diagnostic")


def _first_checkpoint(path: Path) -> dict[str, object]:
    custodians = _custodians()
    snapshot, authorities = _key_status_snapshot(custodians)
    checkpoint, _ = _append(path, snapshot, custodians, authorities)
    return checkpoint


def _signed(
    checkpoint: dict[str, object],
    signer: _CustodianAdapter,
) -> dict[str, object]:
    return create_signed_custody_status_checkpoint(
        checkpoint,
        signer,
        log_id=LOG_ID,
        source_revision=SOURCE_REVISION,
        signed_at_epoch=SIGNED_AT,
    )


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def test_signed_checkpoint_round_trip_is_privacy_reduced(tmp_path: Path) -> None:
    path = tmp_path / "custody-status.log"
    checkpoint = _first_checkpoint(path)
    signer = _CustodianAdapter("checkpoint-signer")
    envelope = _signed(checkpoint, signer)

    payload = serialize_signed_custody_status_checkpoint(envelope)
    loaded = load_signed_custody_status_checkpoint(payload)
    receipt = verify_signed_custody_status_checkpoint(
        loaded,
        signer,
        path,
        expected_log_id=LOG_ID,
        expected_source_revision=SOURCE_REVISION,
    )

    assert loaded == envelope
    assert receipt["schema_version"] == SIGNED_STATUS_CHECKPOINT_VERIFICATION_SCHEMA
    assert receipt["verified"] is True
    assert receipt["retained_tree_size"] == receipt["current_tree_size"] == 1
    assert receipt["retained_checkpoint_sha256"] == checkpoint["checkpoint_sha256"]
    assert len(str(receipt["verification_sha256"])) == 64
    assert signer.key_id not in json.dumps(receipt)
    assert signer.key_revision_sha256 not in json.dumps(receipt)
    assert "signature" not in receipt


def test_signed_checkpoint_proves_later_log_extension(tmp_path: Path) -> None:
    path = tmp_path / "custody-status.log"
    first = _first_checkpoint(path)
    signer = _CustodianAdapter("checkpoint-signer")
    envelope = _signed(first, signer)
    custodians = _custodians()
    second_snapshot, second_authorities = _key_status_snapshot(
        custodians,
        statuses={custodians["custodian-2"].key_id: "revoked"},
        issued_at_epoch=ATTESTED_AT + 1,
    )
    current, appended = _append(
        path,
        second_snapshot,
        custodians,
        second_authorities,
        epoch=VERIFICATION_EPOCH + 1,
    )

    receipt = verify_signed_custody_status_checkpoint(
        envelope,
        signer,
        path,
        expected_log_id=LOG_ID,
        expected_source_revision=SOURCE_REVISION,
    )

    assert appended is True
    assert receipt["retained_tree_size"] == 1
    assert receipt["current_tree_size"] == 2
    assert receipt["current_checkpoint_sha256"] == current["checkpoint_sha256"]


def test_truncation_and_rewritten_history_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "custody-status.log"
    checkpoint = _first_checkpoint(path)
    signer = _CustodianAdapter("checkpoint-signer")
    envelope = _signed(checkpoint, signer)
    original = path.read_bytes()

    path.write_bytes(b"")
    os.chmod(path, 0o600)
    with pytest.raises(SignedCustodyStatusCheckpointError, match="does not extend"):
        verify_signed_custody_status_checkpoint(
            envelope,
            signer,
            path,
            expected_log_id=LOG_ID,
            expected_source_revision=SOURCE_REVISION,
        )

    entry = json.loads(original)
    entry["observed_at_epoch"] = int(entry["observed_at_epoch"]) + 1
    entry["entry_sha256"] = _digest({key: value for key, value in entry.items() if key != "entry_sha256"})
    path.write_bytes(json.dumps(entry, separators=(",", ":"), sort_keys=True).encode("ascii") + b"\n")
    os.chmod(path, 0o600)
    with pytest.raises(SignedCustodyStatusCheckpointError, match="does not extend"):
        verify_signed_custody_status_checkpoint(
            envelope,
            signer,
            path,
            expected_log_id=LOG_ID,
            expected_source_revision=SOURCE_REVISION,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("source_revision", "b" * 40, "digest does not match"),
        ("key_id", "replacement-key", "digest does not match"),
        ("signed_at_epoch", SIGNED_AT + 1, "digest does not match"),
        ("signature", "A" * 86, "digest does not match"),
    ],
)
def test_bound_claim_tampering_is_rejected(
    tmp_path: Path,
    field: str,
    replacement: object,
    message: str,
) -> None:
    path = tmp_path / "custody-status.log"
    signer = _CustodianAdapter("checkpoint-signer")
    envelope = _signed(_first_checkpoint(path), signer)
    tampered = {**envelope, field: replacement}

    with pytest.raises(SignedCustodyStatusCheckpointError, match=message):
        verify_signed_custody_status_checkpoint(
            tampered,
            signer,
            path,
            expected_log_id=LOG_ID,
            expected_source_revision=SOURCE_REVISION,
        )


def test_resigned_substitution_still_rejects_wrong_expectations(tmp_path: Path) -> None:
    path = tmp_path / "custody-status.log"
    signer = _CustodianAdapter("checkpoint-signer")
    checkpoint = _first_checkpoint(path)
    envelope = create_signed_custody_status_checkpoint(
        checkpoint,
        signer,
        log_id=LOG_ID,
        source_revision="b" * 40,
        signed_at_epoch=SIGNED_AT,
    )

    with pytest.raises(SignedCustodyStatusCheckpointError, match="source revision mismatch"):
        verify_signed_custody_status_checkpoint(
            envelope,
            signer,
            path,
            expected_log_id=LOG_ID,
            expected_source_revision=SOURCE_REVISION,
        )
    with pytest.raises(SignedCustodyStatusCheckpointError, match="checkpoint is invalid"):
        verify_signed_custody_status_checkpoint(
            envelope,
            signer,
            path,
            expected_log_id="different-log",
            expected_source_revision="b" * 40,
        )


def test_wrong_and_unavailable_verifiers_fail_without_private_diagnostics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "custody-status.log"
    signer = _CustodianAdapter("checkpoint-signer")
    envelope = _signed(_first_checkpoint(path), signer)

    with pytest.raises(SignedCustodyStatusCheckpointError, match="metadata mismatch"):
        verify_signed_custody_status_checkpoint(
            envelope,
            _CustodianAdapter("wrong-signer"),
            path,
            expected_log_id=LOG_ID,
            expected_source_revision=SOURCE_REVISION,
        )
    unavailable = _UnavailableVerifier("checkpoint-signer")
    with pytest.raises(SignedCustodyStatusCheckpointError) as captured:
        verify_signed_custody_status_checkpoint(
            envelope,
            unavailable,
            path,
            expected_log_id=LOG_ID,
            expected_source_revision=SOURCE_REVISION,
        )

    assert "checkpoint signature verification failed" in str(captured.value)
    assert "private provider diagnostic" not in str(captured.value)


def test_loader_rejects_noncanonical_duplicate_and_oversized_payloads(
    tmp_path: Path,
) -> None:
    path = tmp_path / "custody-status.log"
    signer = _CustodianAdapter("checkpoint-signer")
    envelope = _signed(_first_checkpoint(path), signer)
    payload = serialize_signed_custody_status_checkpoint(envelope)

    with pytest.raises(SignedCustodyStatusCheckpointError, match="not canonical"):
        load_signed_custody_status_checkpoint(payload.replace(b'":', b'": ', 1))
    with pytest.raises(SignedCustodyStatusCheckpointError, match="duplicate field"):
        load_signed_custody_status_checkpoint(payload[:-1] + b',"schema_version":"duplicate"}')
    with pytest.raises(SignedCustodyStatusCheckpointError, match="payload is invalid"):
        load_signed_custody_status_checkpoint(b"x" * (MAX_SIGNED_STATUS_CHECKPOINT_BYTES + 1))


def test_creation_rejects_checkpoint_identity_substitution(tmp_path: Path) -> None:
    path = tmp_path / "custody-status.log"
    signer = _CustodianAdapter("checkpoint-signer")
    checkpoint = copy.deepcopy(_first_checkpoint(path))

    with pytest.raises(SignedCustodyStatusCheckpointError, match="checkpoint is invalid"):
        create_signed_custody_status_checkpoint(
            checkpoint,
            signer,
            log_id="different-log",
            source_revision=SOURCE_REVISION,
            signed_at_epoch=SIGNED_AT,
        )
