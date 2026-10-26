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

## Receipt contract

Receipt schema v5 binds:

- world size and local rank;
- workers per rank and epoch;
- a canonical SHA-256 manifest covering every rank/worker seed commitment;
- an optional canonical distributed-sampler manifest covering every rank's order;
- an optional ordered dataset-content manifest nested into the sampler manifest;
- the existing root seed, model/tokenizer artifacts, backend RNG state, and sweep
  evidence.

The manifest publishes only SHA-256 seed commitments, never raw worker seeds. Every
rank derives the same global manifest while receiving only its own initializer
entries. Cross-seed promotion rejects mixed worker topology or dataset identity,
and online verification can require exact worker, sampler, and dataset manifest
digests plus expected topology.

## Boundaries

This proves that Struct-XAI derived and supplied a deterministic initializer; it
cannot prove that arbitrary callback code attached it to every DataLoader. It does
not seed independent NumPy `Generator` instances, remote services, custom native
extensions, or native-library RNGs. The governed input-plan path supplies both the
initializer and sampler, but arbitrary callback code can still ignore them. Dataset
binding proves only that caller-supplied canonical bytes stayed stable; it does not
prove semantic correctness, provenance, or that callback code consumed those exact
objects. SHA-256 commitments also hide values but do not anonymize low-entropy IDs
or payloads against guessing attacks.

With `persistent_workers=True`, PyTorch does not recreate workers each epoch. Use a
new loader per governed epoch or an explicit worker reseeding protocol; changing the
receipt epoch alone cannot reseed already-running processes.
