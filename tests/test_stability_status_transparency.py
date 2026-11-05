from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from structxai.stability_status_transparency import (
    CustodyStatusTransparencyError,
    append_verified_custody_key_status_snapshot,
    verify_custody_status_transparency_log,
)
from tests.test_stability_reference_custody import (
    ATTESTED_AT,
    VERIFICATION_EPOCH,
    _custodians,
    _key_status_snapshot,
)

LOG_ID = "struct-xai-custody-status-production"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _expected_keys(custodians) -> dict[str, str]:
    return {adapter.key_id: adapter.key_revision_sha256 for adapter in custodians.values()}


def _append(path: Path, snapshot, custodians, authorities, *, epoch=VERIFICATION_EPOCH):
    return append_verified_custody_key_status_snapshot(
        path,
        snapshot,
        _expected_keys(custodians),
        authorities,
        log_id=LOG_ID,
        required_authority_signatures=2,
        verification_epoch=epoch,
    )


def test_append_is_durable_privacy_reduced_and_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "custody-status.log"
    custodians = _custodians()
    snapshot, authorities = _key_status_snapshot(custodians)

    first, appended = _append(path, snapshot, custodians, authorities)
    replay, replay_appended = _append(path, snapshot, custodians, authorities)
    verified = verify_custody_status_transparency_log(path, log_id=LOG_ID)
    payload = path.read_bytes()

    assert appended is True
    assert replay_appended is False
    assert first == replay == verified
    assert first["tree_size"] == 1
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert payload.endswith(b"\n")
    for adapter in (*custodians.values(), *authorities.values()):
        assert adapter.key_id.encode() not in payload
        assert adapter.key_revision_sha256.encode() not in payload
    assert b"signature" not in payload


def test_retained_checkpoint_proves_a_later_log_is_an_extension(tmp_path: Path) -> None:
    path = tmp_path / "custody-status.log"
    custodians = _custodians()
    first_snapshot, first_authorities = _key_status_snapshot(custodians)
    prior, _ = _append(path, first_snapshot, custodians, first_authorities)
    revoked_id = custodians["custodian-3"].key_id
    second_snapshot, second_authorities = _key_status_snapshot(
        custodians,
        statuses={revoked_id: "revoked"},
        issued_at_epoch=ATTESTED_AT + 1,
    )

    current, appended = _append(
        path,
        second_snapshot,
        custodians,
        second_authorities,
        epoch=VERIFICATION_EPOCH + 1,
    )
    consistent = verify_custody_status_transparency_log(
        path,
        log_id=LOG_ID,
        prior_checkpoint=prior,
    )

    assert appended is True
    assert current == consistent
    assert current["tree_size"] == 2
    assert current["head_entry_sha256"] != prior["head_entry_sha256"]


def test_invalid_status_is_rejected_before_log_creation(tmp_path: Path) -> None:
    path = tmp_path / "custody-status.log"
    custodians = _custodians()
    snapshot, authorities = _key_status_snapshot(custodians)
    snapshot["statuses"][0]["status"] = "revoked"

    with pytest.raises(CustodyStatusTransparencyError, match="verification failed"):
        _append(path, snapshot, custodians, authorities)

    assert not path.exists()


def test_checkpoint_detects_truncation_and_rewritten_history(tmp_path: Path) -> None:
    path = tmp_path / "custody-status.log"
    custodians = _custodians()
    first_snapshot, first_authorities = _key_status_snapshot(custodians)
    first, _ = _append(path, first_snapshot, custodians, first_authorities)
    second_snapshot, second_authorities = _key_status_snapshot(
        custodians,
        statuses={custodians["custodian-2"].key_id: "revoked"},
        issued_at_epoch=ATTESTED_AT + 1,
    )
    second, _ = _append(
        path,
        second_snapshot,
        custodians,
        second_authorities,
        epoch=VERIFICATION_EPOCH + 1,
    )
    lines = path.read_bytes().splitlines(keepends=True)

    path.write_bytes(lines[0])
    os.chmod(path, 0o600)
    with pytest.raises(CustodyStatusTransparencyError, match="truncated"):
        verify_custody_status_transparency_log(path, log_id=LOG_ID, prior_checkpoint=second)

    entry = json.loads(lines[0])
    entry["observed_at_epoch"] = int(entry["observed_at_epoch"]) + 1
    claims = {key: value for key, value in entry.items() if key != "entry_sha256"}
    entry["entry_sha256"] = hashlib.sha256(_canonical(claims)).hexdigest()
    path.write_bytes(_canonical(entry) + b"\n")
    os.chmod(path, 0o600)
    with pytest.raises(CustodyStatusTransparencyError, match="inconsistent with prior"):
        verify_custody_status_transparency_log(path, log_id=LOG_ID, prior_checkpoint=first)


def test_concurrent_replay_appends_one_entry(tmp_path: Path) -> None:
    path = tmp_path / "custody-status.log"
    custodians = _custodians()
    snapshot, authorities = _key_status_snapshot(custodians)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda _index: _append(path, snapshot, custodians, authorities),
                range(16),
            )
        )

    checkpoint = verify_custody_status_transparency_log(path, log_id=LOG_ID)
    assert sum(appended for _checkpoint, appended in results) == 1
    assert checkpoint["tree_size"] == 1
    assert all(result[0] == checkpoint for result in results)


def test_partial_kernel_writes_are_completed(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "custody-status.log"
    custodians = _custodians()
    snapshot, authorities = _key_status_snapshot(custodians)
    real_write = os.write
    write_sizes: list[int] = []

    def partial_write(descriptor: int, payload: bytes) -> int:
        size = max(1, len(payload) // 2)
        write_sizes.append(size)
        return real_write(descriptor, payload[:size])

    monkeypatch.setattr(os, "write", partial_write)
    checkpoint, appended = _append(path, snapshot, custodians, authorities)

    assert appended is True
    assert len(write_sizes) > 1
    assert verify_custody_status_transparency_log(path, log_id=LOG_ID) == checkpoint


def test_log_rejects_unsafe_files_and_noncanonical_entries(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe.log"
    unsafe.write_bytes(b"")
    os.chmod(unsafe, 0o644)
    with pytest.raises(CustodyStatusTransparencyError, match="permissions are unsafe"):
        verify_custody_status_transparency_log(unsafe, log_id=LOG_ID)

    target = tmp_path / "target.log"
    target.write_bytes(b"")
    os.chmod(target, 0o600)
    link = tmp_path / "link.log"
    link.symlink_to(target)
    with pytest.raises(CustodyStatusTransparencyError, match="opened safely"):
        verify_custody_status_transparency_log(link, log_id=LOG_ID)

    malformed = tmp_path / "malformed.log"
    malformed.write_bytes(b'{"schema_version":"x", "schema_version":"y"}\n')
    os.chmod(malformed, 0o600)
    with pytest.raises(CustodyStatusTransparencyError, match="duplicate field"):
        verify_custody_status_transparency_log(malformed, log_id=LOG_ID)


def test_checkpoint_and_log_identity_substitution_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "custody-status.log"
    custodians = _custodians()
    snapshot, authorities = _key_status_snapshot(custodians)
    checkpoint, _ = _append(path, snapshot, custodians, authorities)

    with pytest.raises(CustodyStatusTransparencyError, match="log identity mismatch"):
        verify_custody_status_transparency_log(path, log_id="different-log")
    with pytest.raises(CustodyStatusTransparencyError, match="metadata is invalid"):
        verify_custody_status_transparency_log(
            path,
            log_id=LOG_ID,
            prior_checkpoint={**checkpoint, "log_id": "different-log"},
        )
