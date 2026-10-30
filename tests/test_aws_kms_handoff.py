from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from structxai.aws_kms_handoff import (
    AWS_KMS_ADAPTER_SCHEMA,
    AwsKmsHandoffAdapter,
    AwsKmsHandoffError,
)

KEY_ARN = "arn:aws:kms:eu-central-1:123456789012:key/12345678-1234-1234-1234-123456789abc"
KEY_UUID = "12345678-1234-1234-1234-123456789abc"
PUBLIC_KEY = b"0" * 91


class _FakeKmsClient:
    def __init__(self) -> None:
        self.metadata: dict[str, object] = {
            "Arn": KEY_ARN,
            "KeyId": KEY_UUID,
            "Enabled": True,
            "KeyState": "Enabled",
            "KeyUsage": "SIGN_VERIFY",
            "KeyManager": "CUSTOMER",
            "KeySpec": "ECC_NIST_P256",
        }
        self.public: dict[str, object] = {
            "KeyId": KEY_ARN,
            "PublicKey": PUBLIC_KEY,
            "KeySpec": "ECC_NIST_P256",
            "KeyUsage": "SIGN_VERIFY",
            "SigningAlgorithms": ["ECDSA_SHA_256"],
        }
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.sign_response: dict[str, object] | None = None
        self.verify_response: dict[str, object] | None = None
        self.failure: str | None = None

    def _record(self, operation: str, kwargs: dict[str, object]) -> None:
        self.calls.append((operation, kwargs))
        if self.failure == operation:
            raise RuntimeError(f"provider diagnostic from {operation}")

    def describe_key(self, **kwargs: object) -> dict[str, object]:
        self._record("describe_key", kwargs)
        return {"KeyMetadata": dict(self.metadata)}

    def get_public_key(self, **kwargs: object) -> dict[str, object]:
        self._record("get_public_key", kwargs)
        return dict(self.public)

    def sign(self, **kwargs: object) -> dict[str, object]:
        self._record("sign", kwargs)
        if self.sign_response is not None:
            return dict(self.sign_response)
        signature = hashlib.sha512(bytes(kwargs["Message"])).digest()
        return {
            "KeyId": KEY_ARN,
            "Signature": signature,
            "SigningAlgorithm": self.public["SigningAlgorithms"][0],
        }

    def verify(self, **kwargs: object) -> dict[str, object]:
        self._record("verify", kwargs)
        if self.verify_response is not None:
            return dict(self.verify_response)
        expected = hashlib.sha512(bytes(kwargs["Message"])).digest()
        return {
            "KeyId": KEY_ARN,
            "SignatureValid": hmac.compare_digest(bytes(kwargs["Signature"]), expected),
            "SigningAlgorithm": self.public["SigningAlgorithms"][0],
        }


def _adapter(client: _FakeKmsClient, key_ref: str = "alias/struct-xai-handoff"):
    return AwsKmsHandoffAdapter(
        client,
        kms_key_ref=key_ref,
        handoff_key_id="primary-handoff-key",
        algorithm="ECDSA-P256-SHA256",
    )


def test_adapter_resolves_alias_then_uses_canonical_arn_for_crypto() -> None:
    client = _FakeKmsClient()
    adapter = _adapter(client)
    message = b'\x7b"authority_epoch":43\x7d'

    signature = adapter.sign(message)
    assert adapter.verify(message, signature)

    assert adapter.algorithm == "ECDSA-P256-SHA256"
    assert adapter.key_id == "primary-handoff-key"
    assert adapter.key_arn == KEY_ARN
    assert adapter.public_key_sha256 == hashlib.sha256(PUBLIC_KEY).hexdigest()
    assert len(adapter.key_revision_sha256) == 64
    assert client.calls == [
        ("describe_key", {"KeyId": "alias/struct-xai-handoff"}),
        ("get_public_key", {"KeyId": KEY_ARN}),
        (
            "sign",
            {
                "KeyId": KEY_ARN,
                "Message": message,
                "MessageType": "RAW",
                "SigningAlgorithm": "ECDSA_SHA_256",
            },
        ),
        (
            "verify",
            {
                "KeyId": KEY_ARN,
                "Message": message,
                "MessageType": "RAW",
                "Signature": signature,
                "SigningAlgorithm": "ECDSA_SHA_256",
            },
        ),
    ]


def test_revision_is_alias_independent_and_public_key_bound() -> None:
    first_client = _FakeKmsClient()
    second_client = _FakeKmsClient()
    first = _adapter(first_client)
    second = _adapter(second_client, KEY_ARN)

    expected_revision = {
        "schema_version": AWS_KMS_ADAPTER_SCHEMA,
        "provider": "aws-kms",
        "key_arn": KEY_ARN,
        "provider_key_id": KEY_UUID,
        "key_spec": "ECC_NIST_P256",
        "key_usage": "SIGN_VERIFY",
        "signing_algorithm": "ECDSA_SHA_256",
        "public_key_sha256": hashlib.sha256(PUBLIC_KEY).hexdigest(),
    }
    expected_digest = hashlib.sha256(
        json.dumps(
            expected_revision,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    assert first.key_revision_sha256 == second.key_revision_sha256 == expected_digest

    second_client = _FakeKmsClient()
    second_client.public["PublicKey"] = b"1" * 91
    changed = _adapter(second_client)
    assert changed.key_revision_sha256 != first.key_revision_sha256


@pytest.mark.parametrize(
    ("algorithm", "aws_algorithm", "key_spec"),
    [
        ("ECDSA-P256-SHA256", "ECDSA_SHA_256", "ECC_NIST_P256"),
        ("RSASSA-PKCS1-V1_5-SHA256", "RSASSA_PKCS1_V1_5_SHA_256", "RSA_2048"),
        ("RSASSA-PSS-SHA256", "RSASSA_PSS_SHA_256", "RSA_4096"),
    ],
)
def test_reviewed_algorithm_mappings_are_enforced(
    algorithm: str,
    aws_algorithm: str,
    key_spec: str,
) -> None:
    client = _FakeKmsClient()
    client.metadata["KeySpec"] = key_spec
    client.public["KeySpec"] = key_spec
    client.public["SigningAlgorithms"] = [aws_algorithm]
    adapter = AwsKmsHandoffAdapter(
        client,
        kms_key_ref=KEY_ARN,
        handoff_key_id="primary",
        algorithm=algorithm,
    )

    signature = adapter.sign(b"claims")
    assert adapter.verify(b"claims", signature)
    assert client.calls[-2][1]["SigningAlgorithm"] == aws_algorithm
    assert client.calls[-1][1]["SigningAlgorithm"] == aws_algorithm


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("Enabled", False, "AWS_KMS_KEY_NOT_ENABLED"),
        ("KeyState", "PendingImport", "AWS_KMS_KEY_NOT_ENABLED"),
        ("KeyUsage", "ENCRYPT_DECRYPT", "AWS_KMS_KEY_USAGE_INVALID"),
        ("KeyManager", "AWS", "AWS_KMS_KEY_MANAGER_INVALID"),
        ("KeySpec", "ECC_NIST_P384", "AWS_KMS_KEY_SPEC_INVALID"),
    ],
)
def test_adapter_rejects_unsafe_key_metadata(field: str, value: object, error: str) -> None:
    client = _FakeKmsClient()
    client.metadata[field] = value

    with pytest.raises(AwsKmsHandoffError, match=error):
        _adapter(client)


def test_adapter_rejects_unsupported_algorithm_and_bad_logical_id() -> None:
    with pytest.raises(AwsKmsHandoffError, match="AWS_KMS_ALGORITHM_UNSUPPORTED"):
        AwsKmsHandoffAdapter(
            _FakeKmsClient(),
            kms_key_ref=KEY_ARN,
            handoff_key_id="primary",
            algorithm="ED25519",
        )
    with pytest.raises(AwsKmsHandoffError, match="AWS_KMS_HANDOFF_KEY_ID_INVALID"):
        AwsKmsHandoffAdapter(
            _FakeKmsClient(),
            kms_key_ref=KEY_ARN,
            handoff_key_id="alias/contains/slashes",
            algorithm="ECDSA-P256-SHA256",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("KeyId", "arn:aws:kms:eu-central-1:123456789012:key/wrong"),
        ("KeyUsage", "ENCRYPT_DECRYPT"),
        ("KeySpec", "ECC_NIST_P384"),
        ("SigningAlgorithms", ["ECDSA_SHA_384"]),
        ("PublicKey", b"short"),
    ],
)
def test_adapter_rejects_inconsistent_public_key_metadata(field: str, value: object) -> None:
    client = _FakeKmsClient()
    client.public[field] = value

    with pytest.raises(AwsKmsHandoffError, match="AWS_KMS_PUBLIC_KEY_INVALID"):
        _adapter(client)


def test_sign_rejects_mismatched_response_and_normalizes_provider_failure() -> None:
    client = _FakeKmsClient()
    adapter = _adapter(client)
    client.sign_response = {
        "KeyId": KEY_ARN,
        "Signature": b"x" * 64,
        "SigningAlgorithm": "RSASSA_PSS_SHA_256",
    }
    with pytest.raises(AwsKmsHandoffError, match="AWS_KMS_SIGN_RESPONSE_INVALID"):
        adapter.sign(b"claims")

    client.failure = "sign"
    with pytest.raises(AwsKmsHandoffError, match="AWS_KMS_SIGN_FAILED") as error:
        adapter.sign(b"claims")
    assert "provider diagnostic" not in str(error.value)


def test_verify_requires_exact_boolean_and_response_identity() -> None:
    client = _FakeKmsClient()
    adapter = _adapter(client)
    signature = hashlib.sha512(b"claims").digest()

    client.verify_response = {
        "KeyId": KEY_ARN,
        "SignatureValid": False,
        "SigningAlgorithm": "ECDSA_SHA_256",
    }
    assert adapter.verify(b"claims", signature) is False

    client.verify_response["SignatureValid"] = 1
    with pytest.raises(AwsKmsHandoffError, match="AWS_KMS_VERIFY_RESPONSE_INVALID"):
        adapter.verify(b"claims", signature)

    client.verify_response = {
        "KeyId": KEY_UUID,
        "SignatureValid": True,
        "SigningAlgorithm": "RSASSA_PSS_SHA_256",
    }
    with pytest.raises(AwsKmsHandoffError, match="AWS_KMS_VERIFY_RESPONSE_INVALID"):
        adapter.verify(b"claims", signature)


def test_message_and_signature_bounds_fail_before_provider_calls() -> None:
    client = _FakeKmsClient()
    adapter = _adapter(client)
    calls = len(client.calls)

    with pytest.raises(AwsKmsHandoffError, match="AWS_KMS_MESSAGE_INVALID"):
        adapter.sign(b"")
    with pytest.raises(AwsKmsHandoffError, match="AWS_KMS_MESSAGE_INVALID"):
        adapter.verify(b"x" * 4_097, b"y" * 64)
    with pytest.raises(AwsKmsHandoffError, match="AWS_KMS_SIGNATURE_INVALID"):
        adapter.verify(b"claims", b"short")
    assert len(client.calls) == calls


def test_metadata_provider_errors_are_stable_and_content_free() -> None:
    client = _FakeKmsClient()
    client.failure = "describe_key"

    with pytest.raises(AwsKmsHandoffError, match="AWS_KMS_METADATA_UNAVAILABLE") as error:
        _adapter(client)
    assert "provider diagnostic" not in str(error.value)
