from __future__ import annotations

import base64
import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from structxai.stability_status_checkpoint_retention import (
    RETAINED_STATUS_CHECKPOINT_SCHEMA,
    StatusCheckpointRetentionError,
    load_retained_status_checkpoint_receipt,
    retain_verified_status_checkpoint,
    serialize_retained_status_checkpoint_receipt,
)
from tests.test_stability_reference_custody import _CustodianAdapter
from tests.test_stability_status_checkpoint import (
    SOURCE_REVISION,
    _first_checkpoint,
    _signed,
)
from tests.test_stability_status_transparency import LOG_ID

BUCKET = "struct-xai-checkpoints-prod"
OWNER = "123456789012"
KMS_KEY_ARN = "arn:aws:kms:eu-central-1:123456789012:key/11111111-2222-3333-4444-555555555555"
NOW = 1_800_000_100
RETAIN_UNTIL = NOW + 90 * 86_400
VERSION_ID = "3/L4kqtJlcpXroDTDmJ+rmSpXd3QBpU"


class Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get_bucket_versioning(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("versioning", kwargs))
        return {"Status": "Enabled"}

    def get_public_access_block(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("public_access", kwargs))
        return {
            "PublicAccessBlockConfiguration": {
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            }
        }

    def get_object_lock_configuration(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("object_lock", kwargs))
        return {"ObjectLockConfiguration": {"ObjectLockEnabled": "Enabled"}}

    def put_object(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("put", kwargs))
        return {
            "ChecksumSHA256": kwargs["ChecksumSHA256"],
            "ServerSideEncryption": "aws:kms",
            "SSEKMSKeyId": KMS_KEY_ARN,
            "VersionId": VERSION_ID,
        }

    def get_object_retention(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("retention", kwargs))
        return {
            "Retention": {
                "Mode": "COMPLIANCE",
                "RetainUntilDate": datetime.fromtimestamp(RETAIN_UNTIL, tz=UTC),
            }
        }


def retain(client: Client, envelope: dict[str, object], signer: _CustodianAdapter, path: Path):
    return retain_verified_status_checkpoint(
        client,
        envelope,
        signer,
        str(path),
        expected_log_id=LOG_ID,
        expected_source_revision=SOURCE_REVISION,
        bucket=BUCKET,
        key_prefix="struct-xai/custody-status-checkpoints",
        kms_key_arn=KMS_KEY_ARN,
        expected_bucket_owner=OWNER,
        now_epoch=NOW,
        retain_until_epoch=RETAIN_UNTIL,
    )


def fixture(tmp_path: Path):
    path = tmp_path / "custody-status.log"
    signer = _CustodianAdapter("checkpoint-signer")
    envelope = _signed(_first_checkpoint(path), signer)
    return path, signer, envelope


def test_retains_authenticated_checkpoint_as_exact_locked_version(tmp_path: Path) -> None:
    path, signer, envelope = fixture(tmp_path)
    client = Client()

    receipt = retain(client, envelope, signer, path)

    put = dict(client.calls[3][1])
    payload = put["Body"]
    checksum = hashlib.sha256(payload).digest()
    checksum_hex = checksum.hex()
    assert [name for name, _ in client.calls] == [
        "versioning",
        "public_access",
        "object_lock",
        "put",
        "retention",
    ]
    assert put == {
        "Bucket": BUCKET,
        "Key": f"struct-xai/custody-status-checkpoints/{checksum_hex}.json",
        "Body": payload,
        "ContentType": "application/json",
        "ChecksumAlgorithm": "SHA256",
        "ChecksumSHA256": base64.b64encode(checksum).decode("ascii"),
        "IfNoneMatch": "*",
        "ObjectLockMode": "COMPLIANCE",
        "ObjectLockRetainUntilDate": datetime.fromtimestamp(RETAIN_UNTIL, tz=UTC),
        "ServerSideEncryption": "aws:kms",
        "SSEKMSKeyId": KMS_KEY_ARN,
        "ExpectedBucketOwner": OWNER,
        "Metadata": {
            "checkpoint-sha256": envelope["envelope_sha256"],
            "schema-version": RETAINED_STATUS_CHECKPOINT_SCHEMA,
        },
    }
    assert client.calls[4][1]["VersionId"] == VERSION_ID
    assert receipt["checkpoint_sha256"] == envelope["envelope_sha256"]
    assert receipt["checksum_sha256"] == checksum_hex
    assert receipt["version_id"] == VERSION_ID
    assert len(str(receipt["receipt_sha256"])) == 64
    serialized = json.dumps(receipt, sort_keys=True)
    assert BUCKET not in serialized
    assert OWNER not in serialized
    assert KMS_KEY_ARN not in serialized

    payload = serialize_retained_status_checkpoint_receipt(receipt)
    assert load_retained_status_checkpoint_receipt(payload) == receipt


def test_invalid_checkpoint_is_rejected_before_aws_access(tmp_path: Path) -> None:
    path, signer, envelope = fixture(tmp_path)
    client = Client()
    tampered = copy.deepcopy(envelope)
    tampered["signature"] = "A" * len(str(tampered["signature"]))

    with pytest.raises(StatusCheckpointRetentionError, match="verification failed"):
        retain(client, tampered, signer, path)

    assert client.calls == []


@pytest.mark.parametrize(
    ("method", "response"),
    [
        ("get_bucket_versioning", {}),
        (
            "get_public_access_block",
            {"PublicAccessBlockConfiguration": {"BlockPublicAcls": True}},
        ),
        ("get_object_lock_configuration", {}),
    ],
)
def test_insufficient_bucket_controls_stop_before_write(
    tmp_path: Path,
    method: str,
    response: object,
) -> None:
    path, signer, envelope = fixture(tmp_path)
    client = Client()
    setattr(client, method, lambda **_: response)

    with pytest.raises(StatusCheckpointRetentionError, match="controls are insufficient"):
        retain(client, envelope, signer, path)

    assert all(name != "put" for name, _ in client.calls)


@pytest.mark.parametrize(
    "response",
    [
        None,
        {},
        {
            "ChecksumSHA256": "wrong",
            "ServerSideEncryption": "aws:kms",
            "SSEKMSKeyId": KMS_KEY_ARN,
            "VersionId": VERSION_ID,
        },
        {
            "ChecksumSHA256": "valid",
            "ServerSideEncryption": "AES256",
            "SSEKMSKeyId": KMS_KEY_ARN,
            "VersionId": VERSION_ID,
        },
        {
            "ChecksumSHA256": "valid",
            "ServerSideEncryption": "aws:kms",
            "SSEKMSKeyId": "wrong",
            "VersionId": VERSION_ID,
        },
    ],
)
def test_malformed_write_response_fails_closed(tmp_path: Path, response: object) -> None:
    path, signer, envelope = fixture(tmp_path)
    client = Client()
    client.put_object = lambda **_: response

    with pytest.raises(StatusCheckpointRetentionError, match="response is malformed"):
        retain(client, envelope, signer, path)


@pytest.mark.parametrize(
    "retention",
    [
        None,
        {},
        {
            "Mode": "GOVERNANCE",
            "RetainUntilDate": datetime.fromtimestamp(RETAIN_UNTIL, tz=UTC),
        },
        {
            "Mode": "COMPLIANCE",
            "RetainUntilDate": datetime.fromtimestamp(RETAIN_UNTIL - 1, tz=UTC),
        },
        {"Mode": "COMPLIANCE", "RetainUntilDate": "not-a-datetime"},
    ],
)
def test_exact_version_retention_must_match(tmp_path: Path, retention: object) -> None:
    path, signer, envelope = fixture(tmp_path)
    client = Client()
    client.get_object_retention = lambda **_: {"Retention": retention}

    with pytest.raises(StatusCheckpointRetentionError, match="readback is insufficient"):
        retain(client, envelope, signer, path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"bucket": "192.168.1.1"}, "policy is invalid"),
        ({"key_prefix": "../checkpoints"}, "policy is invalid"),
        ({"kms_key_arn": "alias/checkpoints"}, "policy is invalid"),
        ({"expected_bucket_owner": "123"}, "policy is invalid"),
        ({"retain_until_epoch": NOW + 60}, "policy is invalid"),
        ({"retain_until_epoch": NOW + 11 * 366 * 86_400}, "policy is invalid"),
    ],
)
def test_policy_is_bounded_before_verification_or_aws(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    path, signer, envelope = fixture(tmp_path)
    client = Client()
    arguments = {
        "expected_log_id": LOG_ID,
        "expected_source_revision": SOURCE_REVISION,
        "bucket": BUCKET,
        "key_prefix": "struct-xai/custody-status-checkpoints",
        "kms_key_arn": KMS_KEY_ARN,
        "expected_bucket_owner": OWNER,
        "now_epoch": NOW,
        "retain_until_epoch": RETAIN_UNTIL,
    }
    arguments.update(overrides)

    with pytest.raises(StatusCheckpointRetentionError, match=message):
        retain_verified_status_checkpoint(client, envelope, signer, str(path), **arguments)

    assert client.calls == []


def test_provider_failure_is_sanitized(tmp_path: Path) -> None:
    path, signer, envelope = fixture(tmp_path)
    client = Client()

    def fail(**_: object) -> object:
        raise RuntimeError("private AWS request, account, and bucket diagnostics")

    client.put_object = fail
    with pytest.raises(StatusCheckpointRetentionError, match="S3 checkpoint write failed") as exc:
        retain(client, envelope, signer, path)

    assert "private AWS" not in str(exc.value)


def test_receipt_loader_rejects_tampering_duplicates_and_noncanonical_json(
    tmp_path: Path,
) -> None:
    path, signer, envelope = fixture(tmp_path)
    receipt = retain(Client(), envelope, signer, path)
    payload = serialize_retained_status_checkpoint_receipt(receipt)

    tampered = {**receipt, "version_id": "other-version"}
    with pytest.raises(StatusCheckpointRetentionError, match="digest does not match"):
        serialize_retained_status_checkpoint_receipt(tampered)
    with pytest.raises(StatusCheckpointRetentionError, match="duplicate field"):
        load_retained_status_checkpoint_receipt(payload[:-1] + b',"schema_version":"duplicate"}')
    with pytest.raises(StatusCheckpointRetentionError, match="not canonical"):
        load_retained_status_checkpoint_receipt(payload.replace(b'":', b'": ', 1))
