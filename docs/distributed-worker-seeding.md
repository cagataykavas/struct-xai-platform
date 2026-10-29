# Authenticated distributed-worker seeding

Activation-patching callbacks may use a PyTorch `DataLoader` for prompts, control
examples, or calibration batches. Seeding only the parent process does not govern
random augmentations or sampling performed inside worker processes.

`PatchRunSigner.run_with_worker_plan` derives a deterministic seed for every
`(epoch, rank, worker_id)` in a bounded global topology. The callback receives a
picklable `WorkerSeedInitializer` compatible with `DataLoader.worker_init_fn`:

```python
from torch.utils.data import DataLoader

result, receipt = signer.run_with_worker_plan(
    lambda worker_init_fn: run_patch_sweep(
        DataLoader(
            dataset,
            num_workers=4,
            worker_init_fn=worker_init_fn,
            persistent_workers=False,
        )
    ),
    seed=42,
    model_artifact_sha256=model_digest,
    tokenizer_artifact_sha256=tokenizer_digest,
    world_size=2,
    rank=0,
    workers_per_rank=4,
    epoch=3,
)
```

Each worker initializes Python, NumPy's legacy global generator, and CPU Torch.
The full world topology is capped at 4,096 workers. Seed derivation is domain
separated, and the builder rejects any derived collision before the run starts.

## Distributed sample-order provenance

`run_with_input_plan` additionally supplies an immutable rank-local sampler. Its
global manifest commits every rank's exact order without publishing raw sample
indices:

```python
from structxai.dataset_manifest import DatasetExample

dataset_examples = [
    DatasetExample(example_id=row.id, payload=canonical_json(row).encode("utf-8"))
    for row in dataset
]

result, receipt = signer.run_with_input_plan(
    lambda worker_init_fn, sampler: run_patch_sweep(
        DataLoader(
            dataset,
            sampler=sampler,
            shuffle=False,
            num_workers=4,
            worker_init_fn=worker_init_fn,
            persistent_workers=False,
        )
    ),
    seed=42,
    model_artifact_sha256=model_digest,
    tokenizer_artifact_sha256=tokenizer_digest,
    world_size=2,
    rank=0,
    workers_per_rank=4,
    dataset_size=len(dataset),
    dataset_examples=dataset_examples,
    epoch=3,
    shuffle=True,
    drop_last=False,
)
```

The plan uses a domain-separated permutation seed, deterministic rank striding,
and explicit padding or truncation. It records dataset size, epoch, shuffle and
drop-last policy, per-rank counts, and per-rank order digests. When
`dataset_examples` is provided, the sampler manifest also commits the ordered
SHA-256 hashes of each caller-canonicalized example ID and payload. Raw IDs and
payloads never enter the manifest. Plans are capped at one million samples and
1,024 ranks; payloads are capped at 16 MiB per example and 50 GiB in total. The
signer rebuilds both plans around the callback, so in-run sampler or dataset
mutation prevents receipt issuance.

## Strict worker acknowledgments

Use `run_with_acknowledged_input_plan` when promotion requires evidence that every
local worker executed the supplied initializer. The initializer first seeds Python,
NumPy, and Torch, then atomically creates one exclusive acknowledgment record for its
worker ID:

```python
result, receipt, acknowledgment_manifest = signer.run_with_acknowledged_input_plan(
    lambda worker_init_fn, sampler: run_patch_sweep(
        DataLoader(
            dataset,
            sampler=sampler,
            shuffle=False,
            num_workers=4,
            worker_init_fn=worker_init_fn,
            persistent_workers=False,
        )
    ),
    seed=42,
    model_artifact_sha256=model_digest,
    tokenizer_artifact_sha256=tokenizer_digest,
    world_size=2,
    rank=0,
    workers_per_rank=4,
    dataset_size=len(dataset),
    dataset_examples=dataset_examples,
    epoch=3,
)
```

Records live only in a mode-`0700` temporary directory and are written with
exclusive creation, mode `0600`, bounded canonical JSON, and a fresh 256-bit run
challenge. Finalization rejects missing, duplicate, unexpected, non-regular,
oversized, non-canonical, or content-mismatched records. The signed manifest contains
only the challenge digest and worker seed commitments; it never publishes raw seeds.
Temporary files are removed on both success and failure.

## Coordinator aggregation across ranks

Local acknowledgments are necessary but not sufficient for a multi-rank job: a
coordinator must refuse promotion when even one rank is absent. Collect the signed
receipt returned by every rank, keyed by its declared rank, and pass the complete set
into the authenticated cross-seed audit:

```python
audit = audit_authenticated_cross_seed_dose_response(
    runs_by_seed,
    {seed: rank_receipts[seed][0] for seed in runs_by_seed},
    keyring,
    rank_receipt_sets=rank_receipts,
)
```

For each seed, `aggregate_rank_acknowledgments` requires exactly ranks
`0..world_size-1`. It verifies every receipt's signature and freshness, requires a
local-worker acknowledgment manifest, and rejects disagreement in topology, CUDA
backend count, artifact identity, worker-seed plan, sampler plan, dataset identity,
or sweep evidence. Its canonical digest commits the ordered rank receipts and their
acknowledgment digests. Authenticated cross-seed schema v2 commits one such digest per
seed, and requires the separately supplied coordinator receipt to be rank 0 from the
same set. This prevents a valid single-rank receipt from standing in for a complete
distributed run.

### Replay-protected coordinator ingestion

For a promotion boundary that survives coordinator restarts, place each completed
`PatchSweepResult.as_dict()` and every rank's `PatchRunReceipt.as_dict()` in a
versioned envelope:

```json
{
  "schema_version": "struct-xai-rank-receipt-ingestion/v1",
  "promotion_id": "release-2026-10-28",
  "runs": [
    {
      "seed": 11,
      "sweep_result": {"...": "complete PatchSweepResult"},
      "rank_receipts": [
        {"...": "rank 0 PatchRunReceipt"},
        {"...": "rank 1 PatchRunReceipt"}
      ]
    }
  ]
}
```

Supply authorized HMAC keys as base64 values in an environment variable rather
than command-line arguments, then ingest the envelope into a persistent SQLite
ledger:

```bash
export STRUCTXAI_PATCH_RECEIPT_KEYRING_JSON='{"primary":"<base64-key>"}'
struct-xai ingest-rank-receipts promotion.json \
  --ledger state/rank-receipts.sqlite3 \
  --output artifacts/authenticated-audit.json
```

The command performs the existing signature, freshness, complete-rank, shared-input,
and cross-seed policy checks before touching replay state. It then claims the
promotion ID and every receipt digest in one `BEGIN IMMEDIATE` transaction. A receipt
already claimed by another promotion fails with `RANK_RECEIPT_REPLAYED`; rebinding a
promotion ID fails with `PROMOTION_ID_REUSED`. An exact retry of the same envelope is
idempotent and returns the stored v2 audit, allowing recovery if the atomic output
write was interrupted after the ledger commit. Ledger and output files are mode
`0600`; duplicate JSON keys, symlink inputs, unknown fields, and envelopes over 8 MiB
are rejected.

### Backup verification and standby restore

Do not activate a standby coordinator with a raw file copy of a live replay ledger.
Create an online SQLite snapshot and a separate integrity manifest:

```bash
struct-xai backup-rank-ledger \
  --ledger state/rank-receipts.sqlite3 \
  --backup backups/rank-receipts.sqlite3 \
  --manifest backups/rank-receipts.manifest.json

struct-xai verify-rank-ledger-backup \
  --backup backups/rank-receipts.sqlite3 \
  --manifest backups/rank-receipts.manifest.json
```

Backup creation uses SQLite's online backup API, so the snapshot is transactionally
consistent even if the source coordinator remains available. Before publication it
requires ledger schema version 1, the exact replay tables, `PRAGMA integrity_check`,
foreign-key consistency, canonical v2 audit JSON, valid SHA-256 fields, and exact
receipt counts for every promotion. Backup and manifest files are created with mode
`0600`; symlinks, non-regular files, oversized files, existing output targets, unknown
manifest fields, and duplicate JSON keys fail closed.

The manifest commits the complete backup bytes and a logical promotion root. Each
promotion leaf binds a hash of the promotion ID, its envelope and audit digests, its
receipt count, and the ordered receipt-digest manifest. Raw promotion IDs and audit
payloads are not copied into the manifest. Verification recomputes both the physical
and logical commitments.

After verification, restore only into a new standby path:

```bash
struct-xai restore-rank-ledger-backup \
  --backup backups/rank-receipts.sqlite3 \
  --manifest backups/rank-receipts.manifest.json \
  --ledger standby/rank-receipts.sqlite3
```

Restore verifies first, copies and fsyncs through a private temporary file, verifies
the copied SHA-256, and publishes with an exclusive no-overwrite operation. Existing
standby state is never replaced. An exact promotion retry after restore remains
idempotent, while cross-promotion receipt replay remains rejected.

SHA-256 detects accidental corruption or changes relative to the supplied manifest;
it does not authenticate the operator who approved both files. Store or sign the
manifest in a separate trusted control plane. A verified standby still must not run
concurrently with the source coordinator for the same namespace: this workflow is
safe state transfer, not leader election, fencing, or distributed consensus.

### Signed standby handoff and fencing

`structxai.rank_ledger_handoff` adds an authorization layer above backup integrity.
An external controller signs one verified backup for one source controller, standby,
authority epoch, short validity window, and downstream fence acknowledgement:

```python
from structxai.rank_ledger_handoff import (
    create_rank_ledger_handoff,
    restore_authorized_rank_ledger,
)

handoff = create_rank_ledger_handoff(
    backup_path,
    backup_manifest,
    authority_epoch=42,
    source_controller_id="controller-primary",
    standby_id="controller-standby-a",
    downstream_fence_sha256=fence_ack_sha256,
    issued_at_epoch=issued_at,
    expires_at_epoch=issued_at + 300,
    key_id="handoff-primary",
    key=controller_signing_key,
)

result = restore_authorized_rank_ledger(
    backup_path,
    backup_manifest,
    handoff,
    controller_keyring,
    standby_ledger_path,
    required_authority_epoch=authority_store.current_epoch(),
    expected_source_controller_id="controller-primary",
    expected_standby_id="controller-standby-a",
    expected_downstream_fence_sha256=fence_service.current_ack_sha256(),
    now_epoch=clock(),
)
```

The HMAC-SHA256 claims bind the backup-manifest digest, physical backup digest, logical
promotion root, controller identities, authority epoch, fence commitment, key ID, and
validity window. Verification uses constant-time comparisons, supports key rotation by
key ID, and requires at least 32-byte keys. Handoffs live for at most 15 minutes. Unknown
or extra fields, duplicate JSON keys, symlinks, oversized files, weak keys, stale/future
documents, changed backups, or any external-state mismatch fail before the restore path
is created. Restore then repeats backup verification and retains the existing exclusive
no-overwrite publication behavior.

The `required_authority_epoch` and expected fence digest must be read independently from
durable external systems; copying the signed values from the handoff back into those
arguments defeats the control. The module does not allocate epochs, revoke the source,
obtain a downstream fence, persist an epoch checkpoint, elect a leader, or provide
consensus. HMAC authenticates parties that share the controller secret; deployments that
need public verification or signer separation should replace this envelope with an
external KMS-backed asymmetric signature while preserving the same claims.

## Receipt contract

Receipt schema v6 binds:

- world size and local rank;
- workers per rank and epoch;
- a canonical SHA-256 manifest covering every rank/worker seed commitment;
- whether strict local-worker acknowledgments were required and their manifest digest;
- an optional canonical distributed-sampler manifest covering every rank's order;
- an optional ordered dataset-content manifest nested into the sampler manifest;
- the existing root seed, model/tokenizer artifacts, backend RNG state, and sweep
  evidence.

The manifest publishes only SHA-256 seed commitments, never raw worker seeds. Every
rank derives the same global manifest while receiving only its own initializer
entries. Cross-seed promotion rejects mixed worker topology or dataset identity, and
online verification can require exact worker, sampler, and dataset manifest digests
plus expected topology. When rank receipt sets are supplied, promotion also rejects
missing ranks and inconsistent rank-level policy.

## Boundaries

The ordinary worker and input-plan paths prove only that Struct-XAI supplied the
initializer; arbitrary callback code can ignore it. The strict acknowledged path
fails when any expected local worker omits the initializer, but it is process-level
evidence rather than remote attestation: a malicious callback can invoke the callable
outside a DataLoader. It also does not seed independent NumPy `Generator` instances,
remote services, custom native extensions, or native-library RNGs. Dataset binding
proves only that caller-supplied canonical bytes stayed stable; it does not prove
semantic correctness, provenance, or that callback code consumed those exact
objects. SHA-256 commitments also hide values but do not anonymize low-entropy IDs or
payloads against guessing attacks.

Rank aggregation is coordinator-side evidence, not consensus or remote attestation.
It assumes the keyring's signers are authorized, clocks satisfy the freshness policy,
and the coordinator receives all receipts over an authenticated transport. It does
not detect a compromised signer or prove that ranks communicated with each other.
The replay ledger is local coordinator state: operators must place it on durable,
access-controlled storage and must not copy or independently operate multiple ledger
files for the same promotion namespace. SQLite transactionality does not provide
distributed consensus across coordinators.

With `persistent_workers=True`, PyTorch does not recreate workers each epoch. Use a
new loader per governed epoch or an explicit worker reseeding protocol; changing the
receipt epoch alone cannot reseed already-running processes.
