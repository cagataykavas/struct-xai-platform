from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import math
import os
import random
import re
import secrets
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from structxai.artifact_manifest import (
    DEFAULT_ARTIFACT_MANIFEST_POLICY,
    ArtifactManifest,
    ArtifactManifestPolicy,
    build_artifact_manifest,
)
from structxai.patch_dose_response import (
    CrossSeedPolicy,
    DoseResponseAuditPolicy,
    PatchSweepError,
    PatchSweepResult,
    audit_cross_seed_dose_response,
    audit_dose_response,
)

RECEIPT_SCHEMA = "struct-xai-patch-run-receipt/v2"
AUTHENTICATED_AUDIT_SCHEMA = "struct-xai-authenticated-cross-seed/v1"
ALGORITHM = "HMAC-SHA256"
PURPOSE = "activation-patch-run"
MINIMUM_KEY_BYTES = 32
MAX_SEED = 2**63 - 1
DEFAULT_MAX_AGE_SECONDS = 300
DEFAULT_CLOCK_SKEW_SECONDS = 5
DEFAULT_RNG_LOCK_TIMEOUT_SECONDS = 30.0
KEY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_RNG_LOCK = threading.Lock()


class PatchRunReceiptError(PatchSweepError):
    """Stable, non-sensitive failure at the authenticated run boundary."""


@dataclass(frozen=True)
class PatchRunReceipt:
    schema_version: str
    algorithm: str
    purpose: str
    key_id: str
    issued_at_epoch: int
    run_id: str
    seed: int
    model_artifact_sha256: str
    tokenizer_artifact_sha256: str
    cuda_device_count: int
    python_rng_before_sha256: str
    numpy_rng_before_sha256: str
    torch_rng_before_sha256: str
    cuda_rng_before_sha256: str
    python_rng_after_sha256: str
    numpy_rng_after_sha256: str
    torch_rng_after_sha256: str
    cuda_rng_after_sha256: str
    sweep_evidence_sha256: str
    signature: str

    def claims(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("signature")
        return payload

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AuthenticatedSeedRun:
    seed: int
    receipt_sha256: str


@dataclass(frozen=True)
class AuthenticatedCrossSeedAudit:
    schema_version: str
    status: str
    cross_seed_evidence_sha256: str
    model_artifact_sha256: str
    tokenizer_artifact_sha256: str
    seed_runs: tuple[AuthenticatedSeedRun, ...]
    evidence_sha256: str

    def body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "cross_seed_evidence_sha256": self.cross_seed_evidence_sha256,
            "model_artifact_sha256": self.model_artifact_sha256,
            "tokenizer_artifact_sha256": self.tokenizer_artifact_sha256,
            "seed_runs": [asdict(run) for run in self.seed_runs],
        }

    def as_dict(self) -> dict[str, object]:
        return {**self.body(), "evidence_sha256": self.evidence_sha256}


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise PatchRunReceiptError("RECEIPT_NOT_CANONICAL") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _validate_digest(value: object, code: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise PatchRunReceiptError(code)
    return value


def _validate_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        raise PatchRunReceiptError("INVALID_SEED")
    return seed


def _python_rng_digest(state: tuple[Any, ...]) -> str:
    version, internal, gaussian = state
    payload = {
        "version": version,
        "internal": list(internal),
        "gaussian": gaussian,
    }
    return _digest(payload)


def _torch_rng_digest(state: torch.Tensor) -> str:
    if state.device.type != "cpu" or state.dtype != torch.uint8 or state.ndim != 1:
        raise PatchRunReceiptError("TORCH_RNG_STATE_INVALID")
    return hashlib.sha256(bytes(state.tolist())).hexdigest()


def _numpy_rng_digest(state: tuple[Any, ...]) -> str:
    if len(state) != 5 or not isinstance(state[0], str) or not isinstance(state[1], np.ndarray):
        raise PatchRunReceiptError("NUMPY_RNG_STATE_INVALID")
    payload = {
        "algorithm": state[0],
        "keys": state[1].tolist(),
        "position": state[2],
        "has_gaussian": state[3],
        "cached_gaussian": state[4],
    }
    return _digest(payload)


def _cuda_rng_digest(states: tuple[torch.Tensor, ...], device_count: int) -> str:
    if len(states) != device_count:
        raise PatchRunReceiptError("CUDA_RNG_STATE_INVALID")
    return _digest(
        {
            "device_count": device_count,
            "state_sha256": [_torch_rng_digest(state) for state in states],
        }
    )


def _capture_cuda_rng() -> tuple[int, tuple[torch.Tensor, ...]]:
    if not torch.cuda.is_available():
        return 0, ()
    try:
        device_count = torch.cuda.device_count()
        states = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    except (RuntimeError, AssertionError) as exc:
        raise PatchRunReceiptError("CUDA_RNG_STATE_UNAVAILABLE") from exc
    if device_count < 1 or len(states) != device_count:
        raise PatchRunReceiptError("CUDA_RNG_STATE_INVALID")
    for state in states:
        _torch_rng_digest(state)
    return device_count, states


def _restore_rng_states(
    python_state: tuple[Any, ...],
    numpy_state: tuple[Any, ...],
    torch_state: torch.Tensor,
    cuda_states: tuple[torch.Tensor, ...],
) -> None:
    failures: list[Exception] = []
    for restore in (
        lambda: random.setstate(python_state),
        lambda: np.random.set_state(numpy_state),
        lambda: torch.set_rng_state(torch_state),
        lambda: torch.cuda.set_rng_state_all(list(cuda_states)) if cuda_states else None,
    ):
        try:
            restore()
        except (RuntimeError, ValueError, TypeError, AssertionError) as exc:
            failures.append(exc)
    if failures:
        raise PatchRunReceiptError("RNG_STATE_RESTORE_FAILED") from failures[0]


def _validate_key_id(key_id: object) -> str:
    if not isinstance(key_id, str) or KEY_ID_RE.fullmatch(key_id) is None:
        raise PatchRunReceiptError("KEY_ID_INVALID")
    return key_id


def _validate_key(key: object) -> bytes:
    if not isinstance(key, bytes) or len(key) < MINIMUM_KEY_BYTES:
        raise PatchRunReceiptError("SIGNING_KEY_INVALID")
    return key


def _validate_run_id(run_id: object) -> str:
    if not isinstance(run_id, str) or RUN_ID_RE.fullmatch(run_id) is None:
        raise PatchRunReceiptError("SIGNING_IDENTITY_INVALID")
    return run_id


def _issued_at(clock: Callable[[], float]) -> int:
    try:
        value = int(clock())
    except (TypeError, ValueError, OverflowError) as exc:
        raise PatchRunReceiptError("SIGNING_IDENTITY_INVALID") from exc
    if value < 0:
        raise PatchRunReceiptError("SIGNING_IDENTITY_INVALID")
    return value


class PatchRunSigner:
    """Seed a run, restore caller RNG state, and authenticate minimized provenance."""

    def __init__(
        self,
        key_id: str,
        key: bytes,
        *,
        clock: Callable[[], float] = time.time,
        run_id_factory: Callable[[], str] = lambda: secrets.token_hex(16),
        rng_lock_timeout_seconds: float = DEFAULT_RNG_LOCK_TIMEOUT_SECONDS,
    ) -> None:
        self._key_id = _validate_key_id(key_id)
        self._key = _validate_key(key)
        self._clock = clock
        self._run_id_factory = run_id_factory
        if (
            isinstance(rng_lock_timeout_seconds, bool)
            or not isinstance(rng_lock_timeout_seconds, (int, float))
            or not math.isfinite(rng_lock_timeout_seconds)
            or rng_lock_timeout_seconds < 0
        ):
            raise PatchRunReceiptError("RNG_LOCK_POLICY_INVALID")
        self._rng_lock_timeout_seconds = float(rng_lock_timeout_seconds)

    @classmethod
    def from_env(cls) -> PatchRunSigner:
        key_id = os.getenv("STRUCTXAI_PATCH_RECEIPT_KEY_ID")
        encoded_key = os.getenv("STRUCTXAI_PATCH_RECEIPT_KEY_B64")
        if not key_id or not encoded_key:
            raise PatchRunReceiptError("SIGNING_CONFIGURATION_MISSING")
        try:
            key = base64.b64decode(encoded_key, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise PatchRunReceiptError("SIGNING_KEY_INVALID") from exc
        return cls(key_id, key)

    def run(
        self,
        callback: Callable[[], PatchSweepResult],
        *,
        seed: int,
        model_artifact_sha256: str,
        tokenizer_artifact_sha256: str,
    ) -> tuple[PatchSweepResult, PatchRunReceipt]:
        active_seed = _validate_seed(seed)
        model_digest = _validate_digest(model_artifact_sha256, "MODEL_ARTIFACT_DIGEST_INVALID")
        tokenizer_digest = _validate_digest(tokenizer_artifact_sha256, "TOKENIZER_ARTIFACT_DIGEST_INVALID")
        if not callable(callback):
            raise PatchRunReceiptError("RUN_CALLBACK_INVALID")

        if not _RNG_LOCK.acquire(timeout=self._rng_lock_timeout_seconds):
            raise PatchRunReceiptError("RNG_BOUNDARY_BUSY")
        try:
            python_original = random.getstate()
            numpy_original = np.random.get_state()
            torch_original = torch.get_rng_state()
            cuda_device_count, cuda_original = _capture_cuda_rng()
            try:
                random.seed(active_seed)
                np.random.seed(active_seed % (2**32))
                torch.random.default_generator.manual_seed(active_seed)
                if cuda_device_count:
                    torch.cuda.manual_seed_all(active_seed)
                python_before = _python_rng_digest(random.getstate())
                numpy_before = _numpy_rng_digest(np.random.get_state())
                torch_before = _torch_rng_digest(torch.get_rng_state())
                cuda_before = _cuda_rng_digest(_capture_cuda_rng()[1], cuda_device_count)
                result = callback()
                audit_dose_response(result)
                python_after = _python_rng_digest(random.getstate())
                numpy_after = _numpy_rng_digest(np.random.get_state())
                torch_after = _torch_rng_digest(torch.get_rng_state())
                cuda_after_count, cuda_after_states = _capture_cuda_rng()
                if cuda_after_count != cuda_device_count:
                    raise PatchRunReceiptError("CUDA_DEVICE_SET_CHANGED_DURING_RUN")
                cuda_after = _cuda_rng_digest(cuda_after_states, cuda_device_count)
            finally:
                _restore_rng_states(
                    python_original,
                    numpy_original,
                    torch_original,
                    cuda_original,
                )
        finally:
            _RNG_LOCK.release()

        issued_at = _issued_at(self._clock)
        run_id = _validate_run_id(self._run_id_factory())
        claims: dict[str, object] = {
            "schema_version": RECEIPT_SCHEMA,
            "algorithm": ALGORITHM,
            "purpose": PURPOSE,
            "key_id": self._key_id,
            "issued_at_epoch": issued_at,
            "run_id": run_id,
            "seed": active_seed,
            "model_artifact_sha256": model_digest,
            "tokenizer_artifact_sha256": tokenizer_digest,
            "cuda_device_count": cuda_device_count,
            "python_rng_before_sha256": python_before,
            "numpy_rng_before_sha256": numpy_before,
            "torch_rng_before_sha256": torch_before,
            "cuda_rng_before_sha256": cuda_before,
            "python_rng_after_sha256": python_after,
            "numpy_rng_after_sha256": numpy_after,
            "torch_rng_after_sha256": torch_after,
            "cuda_rng_after_sha256": cuda_after,
            "sweep_evidence_sha256": result.evidence_sha256,
        }
        signature = hmac.new(self._key, _canonical_json(claims), hashlib.sha256).hexdigest()
        return result, PatchRunReceipt(**claims, signature=signature)  # type: ignore[arg-type]

    def run_from_artifacts(
        self,
        callback: Callable[[], PatchSweepResult],
        *,
        seed: int,
        model_path: str | Path,
        tokenizer_path: str | Path,
        artifact_policy: ArtifactManifestPolicy = DEFAULT_ARTIFACT_MANIFEST_POLICY,
    ) -> tuple[PatchSweepResult, PatchRunReceipt, ArtifactManifest, ArtifactManifest]:
        """Compute bounded artifact identities and reject mutation across the run."""

        model_manifest = build_artifact_manifest(model_path, artifact_policy)
        tokenizer_manifest = build_artifact_manifest(tokenizer_path, artifact_policy)
        result, receipt = self.run(
            callback,
            seed=seed,
            model_artifact_sha256=model_manifest.evidence_sha256,
            tokenizer_artifact_sha256=tokenizer_manifest.evidence_sha256,
        )
        if build_artifact_manifest(model_path, artifact_policy) != model_manifest:
            raise PatchRunReceiptError("MODEL_ARTIFACT_CHANGED_DURING_RUN")
        if build_artifact_manifest(tokenizer_path, artifact_policy) != tokenizer_manifest:
            raise PatchRunReceiptError("TOKENIZER_ARTIFACT_CHANGED_DURING_RUN")
        return result, receipt, model_manifest, tokenizer_manifest


def verify_patch_run_receipt(
    receipt: PatchRunReceipt,
    keyring: Mapping[str, bytes],
    *,
    expected_seed: int,
    expected_sweep_evidence_sha256: str,
    expected_model_artifact_sha256: str,
    expected_tokenizer_artifact_sha256: str,
    now_epoch: int | None = None,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    clock_skew_seconds: int = DEFAULT_CLOCK_SKEW_SECONDS,
) -> None:
    """Fail closed unless a receipt is authentic, fresh, and bound to this run."""

    if not isinstance(receipt, PatchRunReceipt):
        raise PatchRunReceiptError("RECEIPT_SCHEMA_INVALID")
    if (
        receipt.schema_version != RECEIPT_SCHEMA
        or receipt.algorithm != ALGORITHM
        or receipt.purpose != PURPOSE
        or not isinstance(receipt.run_id, str)
        or RUN_ID_RE.fullmatch(receipt.run_id) is None
        or type(receipt.issued_at_epoch) is not int
        or type(receipt.cuda_device_count) is not int
        or receipt.cuda_device_count < 0
    ):
        raise PatchRunReceiptError("RECEIPT_SCHEMA_INVALID")
    _validate_key_id(receipt.key_id)
    _validate_seed(receipt.seed)
    for value in (
        receipt.model_artifact_sha256,
        receipt.tokenizer_artifact_sha256,
        receipt.python_rng_before_sha256,
        receipt.numpy_rng_before_sha256,
        receipt.torch_rng_before_sha256,
        receipt.cuda_rng_before_sha256,
        receipt.python_rng_after_sha256,
        receipt.numpy_rng_after_sha256,
        receipt.torch_rng_after_sha256,
        receipt.cuda_rng_after_sha256,
        receipt.sweep_evidence_sha256,
        receipt.signature,
    ):
        _validate_digest(value, "RECEIPT_SCHEMA_INVALID")

    if not isinstance(keyring, Mapping):
        raise PatchRunReceiptError("KEYRING_INVALID")
    key = keyring.get(receipt.key_id)
    if key is None:
        raise PatchRunReceiptError("SIGNING_KEY_UNKNOWN")
    expected_signature = hmac.new(
        _validate_key(key), _canonical_json(receipt.claims()), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(receipt.signature, expected_signature):
        raise PatchRunReceiptError("RECEIPT_SIGNATURE_INVALID")

    if (
        receipt.seed != _validate_seed(expected_seed)
        or not hmac.compare_digest(
            receipt.sweep_evidence_sha256,
            _validate_digest(expected_sweep_evidence_sha256, "SWEEP_DIGEST_INVALID"),
        )
        or not hmac.compare_digest(
            receipt.model_artifact_sha256,
            _validate_digest(expected_model_artifact_sha256, "MODEL_ARTIFACT_DIGEST_INVALID"),
        )
        or not hmac.compare_digest(
            receipt.tokenizer_artifact_sha256,
            _validate_digest(
                expected_tokenizer_artifact_sha256,
                "TOKENIZER_ARTIFACT_DIGEST_INVALID",
            ),
        )
    ):
        raise PatchRunReceiptError("RECEIPT_BINDING_MISMATCH")

    if (
        type(max_age_seconds) is not int
        or max_age_seconds < 1
        or type(clock_skew_seconds) is not int
        or clock_skew_seconds < 0
    ):
        raise PatchRunReceiptError("FRESHNESS_POLICY_INVALID")
    now = int(time.time()) if now_epoch is None else now_epoch
    if type(now) is not int or now < 0:
        raise PatchRunReceiptError("VERIFICATION_TIME_INVALID")
    age = now - receipt.issued_at_epoch
    if age < -clock_skew_seconds:
        raise PatchRunReceiptError("RECEIPT_FROM_FUTURE")
    if age > max_age_seconds:
        raise PatchRunReceiptError("RECEIPT_EXPIRED")


def audit_authenticated_cross_seed_dose_response(
    runs: Mapping[int, PatchSweepResult],
    receipts: Mapping[int, PatchRunReceipt],
    keyring: Mapping[str, bytes],
    *,
    now_epoch: int | None = None,
    dose_policy: DoseResponseAuditPolicy | None = None,
    cross_seed_policy: CrossSeedPolicy | None = None,
) -> AuthenticatedCrossSeedAudit:
    """Verify every run receipt and bind them to the cross-seed promotion audit."""

    if not isinstance(runs, Mapping) or not isinstance(receipts, Mapping):
        raise PatchRunReceiptError("AUTHENTICATED_SEED_RUNS_INVALID")
    if set(runs) != set(receipts):
        raise PatchRunReceiptError("RUN_RECEIPT_SET_MISMATCH")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in runs):
        raise PatchRunReceiptError("INVALID_SEED")

    ordered = tuple(sorted(runs))
    authenticated_runs: list[AuthenticatedSeedRun] = []
    artifact_identity: tuple[str, str] | None = None
    cuda_device_count: int | None = None
    for seed in ordered:
        result = runs[seed]
        receipt = receipts[seed]
        verify_patch_run_receipt(
            receipt,
            keyring,
            expected_seed=seed,
            expected_sweep_evidence_sha256=result.evidence_sha256,
            expected_model_artifact_sha256=receipt.model_artifact_sha256,
            expected_tokenizer_artifact_sha256=receipt.tokenizer_artifact_sha256,
            now_epoch=now_epoch,
        )
        current_identity = (
            receipt.model_artifact_sha256,
            receipt.tokenizer_artifact_sha256,
        )
        if artifact_identity is None:
            artifact_identity = current_identity
        elif current_identity != artifact_identity:
            raise PatchRunReceiptError("CROSS_SEED_ARTIFACT_MISMATCH")
        if cuda_device_count is None:
            cuda_device_count = receipt.cuda_device_count
        elif receipt.cuda_device_count != cuda_device_count:
            raise PatchRunReceiptError("CROSS_SEED_RNG_BACKEND_MISMATCH")
        authenticated_runs.append(AuthenticatedSeedRun(seed=seed, receipt_sha256=_digest(receipt.as_dict())))

    if artifact_identity is None:
        raise PatchRunReceiptError("RUN_RECEIPT_SET_MISMATCH")
    audit = audit_cross_seed_dose_response(
        runs,
        dose_policy=dose_policy,
        cross_seed_policy=cross_seed_policy,
    )
    body = {
        "schema_version": AUTHENTICATED_AUDIT_SCHEMA,
        "status": audit.status,
        "cross_seed_evidence_sha256": audit.evidence_sha256,
        "model_artifact_sha256": artifact_identity[0],
        "tokenizer_artifact_sha256": artifact_identity[1],
        "seed_runs": [asdict(run) for run in authenticated_runs],
    }
    return AuthenticatedCrossSeedAudit(
        schema_version=AUTHENTICATED_AUDIT_SCHEMA,
        status=audit.status,
        cross_seed_evidence_sha256=audit.evidence_sha256,
        model_artifact_sha256=artifact_identity[0],
        tokenizer_artifact_sha256=artifact_identity[1],
        seed_runs=tuple(authenticated_runs),
        evidence_sha256=_digest(body),
    )
