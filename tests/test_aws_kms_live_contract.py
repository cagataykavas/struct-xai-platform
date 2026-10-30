from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import ANY

import pytest

from structxai.aws_kms_live_contract import (
    LIVE_KMS_CONTRACT_SCHEMA,
    LiveKmsContractConfig,
    LiveKmsContractError,
    create_bounded_kms_client,
    dump_live_kms_contract_evidence,
    run_live_kms_contract,
    write_live_kms_contract_evidence,
)

KEY_ARN = "arn:aws:kms:eu-central-1:123456789012:key/12345678-1234-1234-1234-123456789abc"
KEY_UUID = "12345678-1234-1234-1234-123456789abc"
PUBLIC_KEY = b"0" * 91


class _FakeKmsClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.negative_result = False

    def _record(self, operation: str, kwargs: dict[str, object]) -> None:
        self.calls.append((operation, kwargs))

    def describe_key(self, **kwargs: object) -> dict[str, object]:
        self._record("describe_key", kwargs)
        return {
            "KeyMetadata": {
                "Arn": KEY_ARN,
                "KeyId": KEY_UUID,
                "Enabled": True,
                "KeyState": "Enabled",
                "KeyUsage": "SIGN_VERIFY",
                "KeyManager": "CUSTOMER",
                "KeySpec": "ECC_NIST_P256",
            }
        }

    def get_public_key(self, **kwargs: object) -> dict[str, object]:
        self._record("get_public_key", kwargs)
        return {
            "KeyId": KEY_ARN,
            "PublicKey": PUBLIC_KEY,
            "KeySpec": "ECC_NIST_P256",
            "KeyUsage": "SIGN_VERIFY",
            "SigningAlgorithms": ["ECDSA_SHA_256"],
        }

    def sign(self, **kwargs: object) -> dict[str, object]:
        self._record("sign", kwargs)
        return {
            "KeyId": KEY_ARN,
            "Signature": hashlib.sha512(bytes(kwargs["Message"])).digest(),
            "SigningAlgorithm": "ECDSA_SHA_256",
        }

    def verify(self, **kwargs: object) -> dict[str, object]:
        self._record("verify", kwargs)
        message = bytes(kwargs["Message"])
        signature = bytes(kwargs["Signature"])
        expected = hashlib.sha512(message).digest()
        result = signature == expected
        if message.endswith(b"\n"):
            result = self.negative_result
        return {
            "KeyId": KEY_ARN,
            "SignatureValid": result,
            "SigningAlgorithm": "ECDSA_SHA_256",
        }


def _config(**changes: object) -> LiveKmsContractConfig:
    values: dict[str, object] = {
        "key_arn": KEY_ARN,
        "region": "eu-central-1",
        "expected_account_id": "123456789012",
        "handoff_key_id": "live-contract-key",
        "algorithm": "ECDSA-P256-SHA256",
    }
    values.update(changes)
    return LiveKmsContractConfig(**values)


def test_live_contract_exercises_positive_and_negative_verification() -> None:
    client = _FakeKmsClient()
    evidence = run_live_kms_contract(
        client,
        _config(),
        probe_id="run-123-1",
        observed_at_epoch=1_798_000_000,
    )

    assert evidence.schema_version == LIVE_KMS_CONTRACT_SCHEMA
    assert evidence.positive_verification is True
    assert evidence.negative_verification_rejected is True
    assert evidence.key_arn_sha256 == hashlib.sha256(KEY_ARN.encode("ascii")).hexdigest()
    assert KEY_ARN not in dump_live_kms_contract_evidence(evidence).decode("ascii")
    assert [operation for operation, _ in client.calls] == [
        "describe_key",
        "get_public_key",
        "sign",
        "verify",
        "verify",
    ]
    assert client.calls[0][1] == {"KeyId": KEY_ARN}
    assert all(
        call["KeyId"] == KEY_ARN
        for operation, call in client.calls[1:]
        if operation in {"get_public_key", "sign", "verify"}
    )


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"key_arn": "alias/struct-xai"}, "LIVE_KMS_KEY_ARN_INVALID"),
        ({"region": "us-east-1"}, "LIVE_KMS_REGION_MISMATCH"),
        ({"expected_account_id": "000000000000"}, "LIVE_KMS_ACCOUNT_MISMATCH"),
        ({"connect_timeout_seconds": 31}, "LIVE_KMS_TIMEOUT_INVALID"),
        ({"total_max_attempts": 4}, "LIVE_KMS_RETRY_POLICY_INVALID"),
    ],
)
def test_config_rejects_mutable_or_cross_boundary_targets(changes: dict[str, object], code: str) -> None:
    with pytest.raises(LiveKmsContractError, match=code):
        _config(**changes)


def test_config_accepts_supported_isolated_partition_region() -> None:
    key_arn = "arn:aws-iso:kms:us-iso-east-1:123456789012:key/12345678-1234-1234-1234-123456789abc"
    config = _config(key_arn=key_arn, region="us-iso-east-1")

    assert config.key_arn == key_arn


def test_contract_fails_when_negative_control_is_accepted() -> None:
    client = _FakeKmsClient()
    client.negative_result = True

    with pytest.raises(LiveKmsContractError, match="LIVE_KMS_NEGATIVE_VERIFY_FAILED"):
        run_live_kms_contract(
            client,
            _config(),
            probe_id="run-123-1",
            observed_at_epoch=1_798_000_000,
        )


def test_bounded_client_factory_applies_exact_transport_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Config:
        def __init__(self, **kwargs: object) -> None:
            captured["config"] = kwargs

    boto3 = ModuleType("boto3")

    def client(service: str, **kwargs: object) -> object:
        captured["client"] = (service, kwargs)
        return SimpleNamespace()

    boto3.client = client  # type: ignore[attr-defined]
    botocore = ModuleType("botocore")
    botocore_config = ModuleType("botocore.config")
    botocore_config.Config = _Config  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "botocore", botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", botocore_config)

    create_bounded_kms_client(_config())

    assert captured["config"] == {
        "connect_timeout": 2.0,
        "read_timeout": 5.0,
        "retries": {"mode": "standard", "total_max_attempts": 2},
    }
    assert captured["client"] == (
        "kms",
        {"region_name": "eu-central-1", "config": ANY},
    )


def test_evidence_write_is_canonical_private_and_atomic(tmp_path: Path) -> None:
    evidence = run_live_kms_contract(
        _FakeKmsClient(),
        _config(),
        probe_id="run-123-1",
        observed_at_epoch=1_798_000_000,
    )
    target = tmp_path / "evidence.json"

    write_live_kms_contract_evidence(target, evidence)

    payload = target.read_bytes()
    assert payload == dump_live_kms_contract_evidence(evidence)
    assert json.loads(payload)["evidence_sha256"] == evidence.evidence_sha256
    assert os.stat(target).st_mode & 0o777 == 0o600
    assert list(tmp_path.iterdir()) == [target]


def test_live_workflow_is_manual_oidc_scoped_and_ephemeral() -> None:
    workflow = Path(".github/workflows/live-kms-contract.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "id-token: write" in workflow
    assert "environment: xai-kms-live" in workflow
    assert "allowed-account-ids:" in workflow
    assert "persist-credentials: false" in workflow
    assert "retention-days: 1" in workflow
    assert "if: always()" in workflow
    assert "rm -f" in workflow
    assert "uses: actions/checkout@v" not in workflow
    assert "uses: actions/setup-python@v" not in workflow
    assert "uses: aws-actions/configure-aws-credentials@v" not in workflow
