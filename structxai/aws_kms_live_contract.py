from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from structxai.aws_kms_handoff import AwsKmsHandoffAdapter, AwsKmsHandoffError

LIVE_KMS_CONTRACT_SCHEMA = "struct-xai-aws-kms-live-contract/v1"
_PURPOSE = "struct-xai-handoff-live-contract"
_KEY_RESOURCE_PATTERN = (
    r"(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"|mrk-[0-9a-f]{32})"
)
_KEY_ARN_RE = re.compile(
    r"^arn:(?P<partition>aws|aws-cn|aws-us-gov|aws-iso|aws-iso-b):kms:"
    rf"(?P<region>[a-z0-9-]+):(?P<account>[0-9]{{12}}):key/{_KEY_RESOURCE_PATTERN}$"
)
_REGION_RE = re.compile(r"[a-z]{2}(?:-[a-z0-9]+)+-[0-9]")
_ACCOUNT_RE = re.compile(r"[0-9]{12}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_ALGORITHMS = {
    "ECDSA-P256-SHA256",
    "RSASSA-PKCS1-V1_5-SHA256",
    "RSASSA-PSS-SHA256",
}


class LiveKmsContractError(RuntimeError):
    """Stable failure from the opt-in live KMS contract boundary."""

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


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _identifier(value: object, code: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise LiveKmsContractError(code)
    return value


@dataclass(frozen=True, slots=True)
class LiveKmsContractConfig:
    key_arn: str
    region: str
    expected_account_id: str
    handoff_key_id: str
    algorithm: str
    connect_timeout_seconds: float = 2.0
    read_timeout_seconds: float = 5.0
    total_max_attempts: int = 2

    def __post_init__(self) -> None:
        match = _KEY_ARN_RE.fullmatch(self.key_arn)
        if match is None:
            raise LiveKmsContractError("LIVE_KMS_KEY_ARN_INVALID")
        if _REGION_RE.fullmatch(self.region) is None or match.group("region") != self.region:
            raise LiveKmsContractError("LIVE_KMS_REGION_MISMATCH")
        if (
            _ACCOUNT_RE.fullmatch(self.expected_account_id) is None
            or match.group("account") != self.expected_account_id
        ):
            raise LiveKmsContractError("LIVE_KMS_ACCOUNT_MISMATCH")
        _identifier(self.handoff_key_id, "LIVE_KMS_HANDOFF_KEY_ID_INVALID")
        if self.algorithm not in _ALGORITHMS:
            raise LiveKmsContractError("LIVE_KMS_ALGORITHM_UNSUPPORTED")
        for value in (self.connect_timeout_seconds, self.read_timeout_seconds):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.1 <= float(value) <= 30.0
            ):
                raise LiveKmsContractError("LIVE_KMS_TIMEOUT_INVALID")
        if type(self.total_max_attempts) is not int or not 1 <= self.total_max_attempts <= 3:
            raise LiveKmsContractError("LIVE_KMS_RETRY_POLICY_INVALID")


@dataclass(frozen=True, slots=True)
class LiveKmsContractEvidence:
    schema_version: str
    provider: str
    purpose: str
    probe_id_sha256: str
    key_arn_sha256: str
    key_revision_sha256: str
    public_key_sha256: str
    algorithm: str
    positive_verification: bool
    negative_verification_rejected: bool
    observed_at_epoch: int
    evidence_sha256: str

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)


def create_bounded_kms_client(config: LiveKmsContractConfig) -> Any:
    """Create a boto3 KMS client with bounded transport and retry behavior."""

    if not isinstance(config, LiveKmsContractConfig):
        raise TypeError("config must be a LiveKmsContractConfig")
    try:
        import boto3
        from botocore.config import Config

        return boto3.client(
            "kms",
            region_name=config.region,
            config=Config(
                connect_timeout=float(config.connect_timeout_seconds),
                read_timeout=float(config.read_timeout_seconds),
                retries={
                    "mode": "standard",
                    "total_max_attempts": config.total_max_attempts,
                },
            ),
        )
    except Exception as exc:
        raise LiveKmsContractError("LIVE_KMS_CLIENT_UNAVAILABLE") from exc


def run_live_kms_contract(
    client: object,
    config: LiveKmsContractConfig,
    *,
    probe_id: str,
    observed_at_epoch: int,
) -> LiveKmsContractEvidence:
    """Exercise positive and negative verification without retaining a signature."""

    if not isinstance(config, LiveKmsContractConfig):
        raise TypeError("config must be a LiveKmsContractConfig")
    probe = _identifier(probe_id, "LIVE_KMS_PROBE_ID_INVALID")
    if type(observed_at_epoch) is not int or observed_at_epoch < 1:
        raise LiveKmsContractError("LIVE_KMS_OBSERVED_AT_INVALID")
    try:
        adapter = AwsKmsHandoffAdapter(
            client,
            kms_key_ref=config.key_arn,
            handoff_key_id=config.handoff_key_id,
            algorithm=config.algorithm,
        )
        if adapter.key_arn != config.key_arn:
            raise LiveKmsContractError("LIVE_KMS_CANONICAL_KEY_MISMATCH")
        claim = {
            "schema_version": LIVE_KMS_CONTRACT_SCHEMA,
            "purpose": _PURPOSE,
            "probe_id": probe,
            "key_revision_sha256": adapter.key_revision_sha256,
        }
        message = _canonical_json(claim)
        signature = adapter.sign(message)
        positive = adapter.verify(message, signature)
        negative = adapter.verify(message + b"\n", signature)
    except LiveKmsContractError:
        raise
    except AwsKmsHandoffError as exc:
        raise LiveKmsContractError(exc.code) from exc
    if positive is not True:
        raise LiveKmsContractError("LIVE_KMS_POSITIVE_VERIFY_FAILED")
    if negative is not False:
        raise LiveKmsContractError("LIVE_KMS_NEGATIVE_VERIFY_FAILED")

    document = {
        "schema_version": LIVE_KMS_CONTRACT_SCHEMA,
        "provider": "aws-kms",
        "purpose": _PURPOSE,
        "probe_id_sha256": hashlib.sha256(probe.encode("ascii")).hexdigest(),
        "key_arn_sha256": hashlib.sha256(config.key_arn.encode("ascii")).hexdigest(),
        "key_revision_sha256": adapter.key_revision_sha256,
        "public_key_sha256": adapter.public_key_sha256,
        "algorithm": config.algorithm,
        "positive_verification": True,
        "negative_verification_rejected": True,
        "observed_at_epoch": observed_at_epoch,
    }
    return LiveKmsContractEvidence(**document, evidence_sha256=_sha256(document))


def dump_live_kms_contract_evidence(evidence: LiveKmsContractEvidence) -> bytes:
    if not isinstance(evidence, LiveKmsContractEvidence):
        raise TypeError("evidence must be LiveKmsContractEvidence")
    document = evidence.to_audit_dict()
    claimed = document.pop("evidence_sha256")
    if claimed != _sha256(document):
        raise LiveKmsContractError("LIVE_KMS_EVIDENCE_DIGEST_MISMATCH")
    document["evidence_sha256"] = claimed
    return _canonical_json(document) + b"\n"


def write_live_kms_contract_evidence(path: str | Path, evidence: LiveKmsContractEvidence) -> None:
    """Atomically persist only the sanitized evidence document with mode 0600."""

    target = Path(path)
    if not target.parent.is_dir():
        raise LiveKmsContractError("LIVE_KMS_EVIDENCE_DIRECTORY_INVALID")
    payload = dump_live_kms_contract_evidence(evidence)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    except OSError as exc:
        raise LiveKmsContractError("LIVE_KMS_EVIDENCE_WRITE_FAILED") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the opt-in AWS KMS live contract")
    parser.add_argument("--key-arn", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--expected-account-id", required=True)
    parser.add_argument("--handoff-key-id", required=True)
    parser.add_argument("--algorithm", required=True, choices=sorted(_ALGORITHMS))
    parser.add_argument("--probe-id", required=True)
    parser.add_argument("--observed-at-epoch", required=True, type=int)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        config = LiveKmsContractConfig(
            key_arn=arguments.key_arn,
            region=arguments.region,
            expected_account_id=arguments.expected_account_id,
            handoff_key_id=arguments.handoff_key_id,
            algorithm=arguments.algorithm,
        )
        evidence = run_live_kms_contract(
            create_bounded_kms_client(config),
            config,
            probe_id=arguments.probe_id,
            observed_at_epoch=arguments.observed_at_epoch,
        )
        write_live_kms_contract_evidence(arguments.output, evidence)
    except LiveKmsContractError as exc:
        print(exc.code, file=sys.stderr)
        return 2
    print(evidence.evidence_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
