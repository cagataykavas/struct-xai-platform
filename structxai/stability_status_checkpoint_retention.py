from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from structxai.stability_manifest import ExternalEvidenceVerifier
from structxai.stability_status_checkpoint import (
    SignedCustodyStatusCheckpointError,
    serialize_signed_custody_status_checkpoint,
    verify_signed_custody_status_checkpoint,
)

RETAINED_STATUS_CHECKPOINT_SCHEMA = "struct-xai-retained-custody-status-checkpoint/v1"
MAX_RETAINED_STATUS_CHECKPOINT_RECEIPT_BYTES = 16 * 1024

_RETENTION_MODE = "COMPLIANCE"
_MIN_RETENTION_SECONDS = 86_400
_MAX_RETENTION_SECONDS = 10 * 366 * 86_400
_BUCKET_RE = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]")
_PREFIX_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,254}")
_OWNER_RE = re.compile(r"[0-9]{12}")
_KMS_KEY_ARN_RE = re.compile(
    r"arn:(?:aws|aws-us-gov|aws-cn):kms:[a-z0-9-]{1,32}:[0-9]{12}:"
    r"key/[A-Za-z0-9-]{1,128}"
)
_VERSION_ID_RE = re.compile(r"[A-Za-z0-9._=+/-]{1,1024}")
_OBJECT_KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/+=-]{0,1023}")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_RECEIPT_FIELDS = frozenset(
    {
        "bucket_sha256",
        "checkpoint_sha256",
        "checksum_sha256",
        "expected_bucket_owner_sha256",
        "kms_key_arn_sha256",
        "object_key",
        "receipt_sha256",
        "retain_until_epoch",
        "retention_mode",
        "schema_version",
        "verification_sha256",
        "version_id",
    }
)


class StatusCheckpointRetentionError(RuntimeError):
    """A signed status checkpoint could not be retained under the reviewed policy."""

    code = "STATUS_CHECKPOINT_RETENTION_REJECTED"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


def retain_verified_status_checkpoint(
    client: Any,
    envelope: Mapping[str, object],
    verifier: ExternalEvidenceVerifier,
    log_path: str,
    *,
    expected_log_id: str,
    expected_source_revision: str,
    bucket: str,
    key_prefix: str,
    kms_key_arn: str,
    expected_bucket_owner: str,
    now_epoch: int,
    retain_until_epoch: int,
) -> dict[str, object]:
    """Authenticate and retain one exact checkpoint as a locked S3 object version."""
    _validate_policy(
        bucket=bucket,
        key_prefix=key_prefix,
        kms_key_arn=kms_key_arn,
        expected_bucket_owner=expected_bucket_owner,
        now_epoch=now_epoch,
        retain_until_epoch=retain_until_epoch,
    )
    try:
        verification = verify_signed_custody_status_checkpoint(
            envelope,
            verifier,
            log_path,
            expected_log_id=expected_log_id,
            expected_source_revision=expected_source_revision,
        )
        payload = serialize_signed_custody_status_checkpoint(envelope)
    except (SignedCustodyStatusCheckpointError, OSError, TypeError, ValueError) as exc:
        raise StatusCheckpointRetentionError("signed checkpoint verification failed") from exc

    _verify_bucket_controls(client, bucket, expected_bucket_owner)
    checksum = hashlib.sha256(payload).digest()
    checksum_base64 = base64.b64encode(checksum).decode("ascii")
    checksum_hex = checksum.hex()
    object_key = f"{key_prefix}/{checksum_hex}.json"
    retain_until = datetime.fromtimestamp(retain_until_epoch, tz=UTC)
    try:
        response = client.put_object(
            Bucket=bucket,
            Key=object_key,
            Body=payload,
            ContentType="application/json",
            ChecksumAlgorithm="SHA256",
            ChecksumSHA256=checksum_base64,
            IfNoneMatch="*",
            ObjectLockMode=_RETENTION_MODE,
            ObjectLockRetainUntilDate=retain_until,
            ServerSideEncryption="aws:kms",
            SSEKMSKeyId=kms_key_arn,
            ExpectedBucketOwner=expected_bucket_owner,
            Metadata={
                "checkpoint-sha256": str(envelope["envelope_sha256"]),
                "schema-version": RETAINED_STATUS_CHECKPOINT_SCHEMA,
            },
        )
    except Exception as exc:
        raise StatusCheckpointRetentionError("S3 checkpoint write failed") from exc
    version_id = _validate_write_response(
        response,
        checksum_base64=checksum_base64,
        kms_key_arn=kms_key_arn,
    )
    _verify_exact_retention(
        client,
        bucket=bucket,
        object_key=object_key,
        version_id=version_id,
        expected_bucket_owner=expected_bucket_owner,
        retain_until=retain_until,
    )
    receipt: dict[str, object] = {
        "schema_version": RETAINED_STATUS_CHECKPOINT_SCHEMA,
        "checkpoint_sha256": str(envelope["envelope_sha256"]),
        "verification_sha256": str(verification["verification_sha256"]),
        "checksum_sha256": checksum_hex,
        "bucket_sha256": _text_digest(bucket),
        "object_key": object_key,
        "version_id": version_id,
        "retention_mode": _RETENTION_MODE,
        "retain_until_epoch": retain_until_epoch,
        "kms_key_arn_sha256": _text_digest(kms_key_arn),
        "expected_bucket_owner_sha256": _text_digest(expected_bucket_owner),
    }
    receipt["receipt_sha256"] = _digest(receipt)
    _validate_receipt(receipt)
    return receipt


def serialize_retained_status_checkpoint_receipt(receipt: Mapping[str, object]) -> bytes:
    """Serialize one exact retention receipt as bounded canonical JSON."""
    normalized = _validate_receipt(receipt)
    payload = _canonical_json(normalized)
    if len(payload) > MAX_RETAINED_STATUS_CHECKPOINT_RECEIPT_BYTES:
        raise StatusCheckpointRetentionError("retention receipt exceeds the size limit")
    return payload


def load_retained_status_checkpoint_receipt(payload: bytes) -> dict[str, object]:
    """Load a canonical retention receipt without accepting ambiguous JSON."""
    if (
        not isinstance(payload, bytes)
        or not 1 <= len(payload) <= MAX_RETAINED_STATUS_CHECKPOINT_RECEIPT_BYTES
    ):
        raise StatusCheckpointRetentionError("retention receipt payload is invalid")
    try:
        value = json.loads(payload, object_pairs_hook=_reject_duplicate_fields)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise StatusCheckpointRetentionError("retention receipt JSON is invalid") from exc
    normalized = _validate_receipt(value)
    if payload != _canonical_json(normalized):
        raise StatusCheckpointRetentionError("retention receipt JSON is not canonical")
    return normalized


def _verify_bucket_controls(client: Any, bucket: str, owner: str) -> None:
    try:
        versioning = client.get_bucket_versioning(
            Bucket=bucket,
            ExpectedBucketOwner=owner,
        )
        public_access = client.get_public_access_block(
            Bucket=bucket,
            ExpectedBucketOwner=owner,
        )
        object_lock = client.get_object_lock_configuration(
            Bucket=bucket,
            ExpectedBucketOwner=owner,
        )
    except Exception as exc:
        raise StatusCheckpointRetentionError("S3 bucket control verification failed") from exc
    configuration = (
        public_access.get("PublicAccessBlockConfiguration", {}) if isinstance(public_access, Mapping) else {}
    )
    object_lock_configuration = (
        object_lock.get("ObjectLockConfiguration", {}) if isinstance(object_lock, Mapping) else {}
    )
    if (
        not isinstance(versioning, Mapping)
        or versioning.get("Status") != "Enabled"
        or not isinstance(configuration, Mapping)
        or any(
            configuration.get(field) is not True
            for field in (
                "BlockPublicAcls",
                "IgnorePublicAcls",
                "BlockPublicPolicy",
                "RestrictPublicBuckets",
            )
        )
        or not isinstance(object_lock_configuration, Mapping)
        or object_lock_configuration.get("ObjectLockEnabled") != "Enabled"
    ):
        raise StatusCheckpointRetentionError("S3 bucket controls are insufficient")


def _validate_write_response(
    response: object,
    *,
    checksum_base64: str,
    kms_key_arn: str,
) -> str:
    if not isinstance(response, Mapping):
        raise StatusCheckpointRetentionError("S3 checkpoint write response is malformed")
    version_id = response.get("VersionId")
    if (
        response.get("ChecksumSHA256") != checksum_base64
        or response.get("ServerSideEncryption") != "aws:kms"
        or response.get("SSEKMSKeyId") != kms_key_arn
        or not isinstance(version_id, str)
        or _VERSION_ID_RE.fullmatch(version_id) is None
    ):
        raise StatusCheckpointRetentionError("S3 checkpoint write response is malformed")
    return version_id


def _verify_exact_retention(
    client: Any,
    *,
    bucket: str,
    object_key: str,
    version_id: str,
    expected_bucket_owner: str,
    retain_until: datetime,
) -> None:
    try:
        response = client.get_object_retention(
            Bucket=bucket,
            Key=object_key,
            VersionId=version_id,
            ExpectedBucketOwner=expected_bucket_owner,
        )
    except Exception as exc:
        raise StatusCheckpointRetentionError("S3 retention readback failed") from exc
    retention = response.get("Retention", {}) if isinstance(response, Mapping) else {}
    observed_until = retention.get("RetainUntilDate") if isinstance(retention, Mapping) else None
    if (
        not isinstance(retention, Mapping)
        or retention.get("Mode") != _RETENTION_MODE
        or not isinstance(observed_until, datetime)
        or observed_until.tzinfo is None
        or observed_until.astimezone(UTC) != retain_until
    ):
        raise StatusCheckpointRetentionError("S3 retention readback is insufficient")


def _validate_policy(
    *,
    bucket: str,
    key_prefix: str,
    kms_key_arn: str,
    expected_bucket_owner: str,
    now_epoch: int,
    retain_until_epoch: int,
) -> None:
    bucket_valid = _valid_bucket(bucket)
    if (
        not bucket_valid
        or not isinstance(key_prefix, str)
        or _PREFIX_RE.fullmatch(key_prefix) is None
        or ".." in key_prefix
        or "//" in key_prefix
        or key_prefix.endswith("/")
        or not isinstance(kms_key_arn, str)
        or _KMS_KEY_ARN_RE.fullmatch(kms_key_arn) is None
        or not isinstance(expected_bucket_owner, str)
        or _OWNER_RE.fullmatch(expected_bucket_owner) is None
        or type(now_epoch) is not int
        or type(retain_until_epoch) is not int
        or now_epoch < 1
        or retain_until_epoch > 253_402_300_799
        or not _MIN_RETENTION_SECONDS <= retain_until_epoch - now_epoch <= _MAX_RETENTION_SECONDS
    ):
        raise StatusCheckpointRetentionError("checkpoint retention policy is invalid")


def _valid_bucket(value: object) -> bool:
    if (
        not isinstance(value, str)
        or _BUCKET_RE.fullmatch(value) is None
        or ".." in value
        or ".-" in value
        or "-." in value
    ):
        return False
    try:
        octets = [int(part) for part in value.split(".")]
    except ValueError:
        return True
    return not (len(octets) == 4 and all(0 <= part <= 255 for part in octets))


def _validate_receipt(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _RECEIPT_FIELDS:
        raise StatusCheckpointRetentionError("retention receipt schema is invalid")
    receipt = dict(value)
    if (
        receipt.get("schema_version") != RETAINED_STATUS_CHECKPOINT_SCHEMA
        or receipt.get("retention_mode") != _RETENTION_MODE
        or not isinstance(receipt.get("object_key"), str)
        or _OBJECT_KEY_RE.fullmatch(str(receipt["object_key"])) is None
        or ".." in str(receipt["object_key"])
        or not isinstance(receipt.get("version_id"), str)
        or _VERSION_ID_RE.fullmatch(str(receipt["version_id"])) is None
        or type(receipt.get("retain_until_epoch")) is not int
        or not 1 <= int(receipt["retain_until_epoch"]) <= 253_402_300_799
        or any(
            not isinstance(receipt.get(field), str) or _DIGEST_RE.fullmatch(str(receipt[field])) is None
            for field in _RECEIPT_FIELDS
            - {
                "schema_version",
                "object_key",
                "version_id",
                "retention_mode",
                "retain_until_epoch",
            }
        )
    ):
        raise StatusCheckpointRetentionError("retention receipt fields are invalid")
    claimed = str(receipt["receipt_sha256"])
    unsigned = {key: receipt[key] for key in _RECEIPT_FIELDS if key != "receipt_sha256"}
    if not hmac.compare_digest(claimed, _digest(unsigned)):
        raise StatusCheckpointRetentionError("retention receipt digest does not match")
    return receipt


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


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
        raise StatusCheckpointRetentionError("retention evidence is not canonical JSON") from exc


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StatusCheckpointRetentionError("retention receipt contains a duplicate field")
        result[key] = value
    return result
