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

## Receipt contract

Receipt schema v3 binds:

- world size and local rank;
- workers per rank and epoch;
- a canonical SHA-256 manifest covering every rank/worker seed commitment;
- the existing root seed, model/tokenizer artifacts, backend RNG state, and sweep
  evidence.

The manifest publishes only SHA-256 seed commitments, never raw worker seeds. Every
rank derives the same global manifest while receiving only its own initializer
entries. Cross-seed promotion rejects mixed worker topology, and online verification
can require both the exact manifest digest and expected topology.

## Boundaries

This proves that Struct-XAI derived and supplied a deterministic initializer; it
cannot prove that arbitrary callback code attached it to every DataLoader. It does
not seed independent NumPy `Generator` instances, remote services, custom native
extensions, or a distributed sampler. DataLoader shuffling and
`DistributedSampler.set_epoch()` remain explicit caller responsibilities.

With `persistent_workers=True`, PyTorch does not recreate workers each epoch. Use a
new loader per governed epoch or an explicit worker reseeding protocol; changing the
receipt epoch alone cannot reseed already-running processes.
