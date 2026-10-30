from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Protocol

from structxai.rank_ledger_handoff import (
    MAX_EXTERNAL_SIGNATURE_BYTES,
    MIN_EXTERNAL_SIGNATURE_BYTES,
)

AWS_KMS_ADAPTER_SCHEMA = "struct-xai-aws-kms-handoff-adapter/v1"
_MAX_RAW_MESSAGE_BYTES = 4_096
_MIN_PUBLIC_KEY_BYTES = 64
_MAX_PUBLIC_KEY_BYTES = 16_384
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_KEY_RESOURCE_PATTERN = r"(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|mrk-[0-9a-f]{32})"
_KEY_ARN_RE = re.compile(
    r"^arn:(?:aws|aws-cn|aws-us-gov|aws-iso|aws-iso-b):kms:"
    rf"[a-z0-9-]+:[0-9]{{12}}:key/{_KEY_RESOURCE_PATTERN}$"
)
_KEY_ID_RE = re.compile(rf"^{_KEY_RESOURCE_PATTERN}$")
_ALGORITHMS = {
    "ECDSA-P256-SHA256": ("ECDSA_SHA_256", frozenset({"ECC_NIST_P256"})),
    "RSASSA-PKCS1-V1_5-SHA256": (
        "RSASSA_PKCS1_V1_5_SHA_256",
        frozenset({"RSA_2048", "RSA_3072", "RSA_4096"}),
    ),
    "RSASSA-PSS-SHA256": (
        "RSASSA_PSS_SHA_256",
        frozenset({"RSA_2048", "RSA_3072", "RSA_4096"}),
    ),
}


class AwsKmsClient(Protocol):
    """Subset of the boto3 KMS client used by the adapter."""

    def describe_key(self, **kwargs: object) -> Mapping[str, object]: ...

    def get_public_key(self, **kwargs: object) -> Mapping[str, object]: ...

    def sign(self, **kwargs: object) -> Mapping[str, object]: ...

    def verify(self, **kwargs: object) -> Mapping[str, object]: ...


class AwsKmsHandoffError(RuntimeError):
    """A stable AWS KMS adapter failure that omits provider diagnostics."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _mapping(value: object, code: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AwsKmsHandoffError(code)
    return value


def _string(value: object, code: str) -> str:
    if not isinstance(value, str):
        raise AwsKmsHandoffError(code)
    return value


def _call(client: AwsKmsClient, method: str, code: str, **kwargs: object) -> Mapping[str, object]:
    try:
        operation = getattr(client, method)
        result = operation(**kwargs)
    except Exception as exc:
        raise AwsKmsHandoffError(code) from exc
    return _mapping(result, code)


class AwsKmsHandoffAdapter:
    """AWS KMS signer/verifier with canonical key and response binding."""

    def __init__(
        self,
        client: AwsKmsClient,
        *,
        kms_key_ref: str,
        handoff_key_id: str,
        algorithm: str,
    ) -> None:
        if not isinstance(kms_key_ref, str) or not kms_key_ref:
            raise AwsKmsHandoffError("AWS_KMS_KEY_REFERENCE_INVALID")
        if not isinstance(handoff_key_id, str) or _IDENTIFIER_RE.fullmatch(handoff_key_id) is None:
            raise AwsKmsHandoffError("AWS_KMS_HANDOFF_KEY_ID_INVALID")
        specification = _ALGORITHMS.get(algorithm)
        if specification is None:
            raise AwsKmsHandoffError("AWS_KMS_ALGORITHM_UNSUPPORTED")
        aws_algorithm, allowed_key_specs = specification

        description = _call(
            client,
            "describe_key",
            "AWS_KMS_METADATA_UNAVAILABLE",
            KeyId=kms_key_ref,
        )
        metadata = _mapping(description.get("KeyMetadata"), "AWS_KMS_METADATA_INVALID")
        key_arn = _string(metadata.get("Arn"), "AWS_KMS_METADATA_INVALID")
        provider_key_id = _string(metadata.get("KeyId"), "AWS_KMS_METADATA_INVALID")
        key_spec = _string(metadata.get("KeySpec"), "AWS_KMS_METADATA_INVALID")
        if _KEY_ARN_RE.fullmatch(key_arn) is None or _KEY_ID_RE.fullmatch(provider_key_id) is None:
            raise AwsKmsHandoffError("AWS_KMS_METADATA_INVALID")
        if metadata.get("Enabled") is not True or metadata.get("KeyState") != "Enabled":
            raise AwsKmsHandoffError("AWS_KMS_KEY_NOT_ENABLED")
        if metadata.get("KeyUsage") != "SIGN_VERIFY":
            raise AwsKmsHandoffError("AWS_KMS_KEY_USAGE_INVALID")
        if metadata.get("KeyManager") != "CUSTOMER":
            raise AwsKmsHandoffError("AWS_KMS_KEY_MANAGER_INVALID")
        if key_spec not in allowed_key_specs:
            raise AwsKmsHandoffError("AWS_KMS_KEY_SPEC_INVALID")

        public = _call(
            client,
            "get_public_key",
            "AWS_KMS_PUBLIC_KEY_UNAVAILABLE",
            KeyId=key_arn,
        )
        public_key = public.get("PublicKey")
        signing_algorithms = public.get("SigningAlgorithms")
        if (
            public.get("KeyId") not in {key_arn, provider_key_id}
            or public.get("KeyUsage") != "SIGN_VERIFY"
            or public.get("KeySpec") != key_spec
            or not isinstance(signing_algorithms, list)
            or any(not isinstance(item, str) for item in signing_algorithms)
            or aws_algorithm not in signing_algorithms
            or not isinstance(public_key, bytes)
            or not _MIN_PUBLIC_KEY_BYTES <= len(public_key) <= _MAX_PUBLIC_KEY_BYTES
        ):
            raise AwsKmsHandoffError("AWS_KMS_PUBLIC_KEY_INVALID")

        public_key_sha256 = hashlib.sha256(public_key).hexdigest()
        revision = {
            "schema_version": AWS_KMS_ADAPTER_SCHEMA,
            "provider": "aws-kms",
            "key_arn": key_arn,
            "provider_key_id": provider_key_id,
            "key_spec": key_spec,
            "key_usage": "SIGN_VERIFY",
            "signing_algorithm": aws_algorithm,
            "public_key_sha256": public_key_sha256,
        }
        self._client = client
        self._key_arn = key_arn
        self._provider_key_id = provider_key_id
        self._aws_algorithm = aws_algorithm
        self.algorithm = algorithm
        self.key_id = handoff_key_id
        self.key_revision_sha256 = hashlib.sha256(_canonical_json(revision)).hexdigest()
        self.public_key_sha256 = public_key_sha256

    @property
    def key_arn(self) -> str:
        return self._key_arn

    def sign(self, message: bytes) -> bytes:
        if not isinstance(message, bytes) or not 1 <= len(message) <= _MAX_RAW_MESSAGE_BYTES:
            raise AwsKmsHandoffError("AWS_KMS_MESSAGE_INVALID")
        response = _call(
            self._client,
            "sign",
            "AWS_KMS_SIGN_FAILED",
            KeyId=self._key_arn,
            Message=message,
            MessageType="RAW",
            SigningAlgorithm=self._aws_algorithm,
        )
        signature = response.get("Signature")
        if (
            response.get("KeyId") not in {self._key_arn, self._provider_key_id}
            or response.get("SigningAlgorithm") != self._aws_algorithm
            or not isinstance(signature, bytes)
            or not MIN_EXTERNAL_SIGNATURE_BYTES <= len(signature) <= MAX_EXTERNAL_SIGNATURE_BYTES
        ):
            raise AwsKmsHandoffError("AWS_KMS_SIGN_RESPONSE_INVALID")
        return signature

    def verify(self, message: bytes, signature: bytes) -> bool:
        if not isinstance(message, bytes) or not 1 <= len(message) <= _MAX_RAW_MESSAGE_BYTES:
            raise AwsKmsHandoffError("AWS_KMS_MESSAGE_INVALID")
        if (
            not isinstance(signature, bytes)
            or not MIN_EXTERNAL_SIGNATURE_BYTES <= len(signature) <= MAX_EXTERNAL_SIGNATURE_BYTES
        ):
            raise AwsKmsHandoffError("AWS_KMS_SIGNATURE_INVALID")
        response = _call(
            self._client,
            "verify",
            "AWS_KMS_VERIFY_FAILED",
            KeyId=self._key_arn,
            Message=message,
            MessageType="RAW",
            Signature=signature,
            SigningAlgorithm=self._aws_algorithm,
        )
        if (
            response.get("KeyId") not in {self._key_arn, self._provider_key_id}
            or response.get("SigningAlgorithm") != self._aws_algorithm
            or type(response.get("SignatureValid")) is not bool
        ):
            raise AwsKmsHandoffError("AWS_KMS_VERIFY_RESPONSE_INVALID")
        return response["SignatureValid"] is True
