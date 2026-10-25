# Bounded activation-patch dose–response sweep

A single full-strength activation patch can produce a striking score change without showing whether
the effect is stable across intervention strength, whether the zero-dose control reproduces the
baseline, or whether a forward hook leaked into later model calls.

`structxai.patch_dose_response` runs an explicit residual-stream interpolation:

```text
patched_state = target_state + alpha * (source_state - target_state)
```

It captures all governed source and target layer states once, then evaluates a strictly increasing
alpha ladder at every requested layer. The runtime:

- requires the ladder to include `0.0` and `1.0` with at least one intermediate dose;
- scores the existing first-token signed candidate margin at every point;
- reports raw margin change and source-normalized recovery;
- reconciles every zero-dose point with the unpatched target baseline;
- repeats the target baseline after the sweep to detect stochastic drift or hook leakage;
- removes hooks in `finally` blocks, including when a model call raises;
- restores the model's original training/evaluation state;
- uses `torch.no_grad()` and never mutates model parameters;
- bounds layers, doses, model calls, sequence length and hidden width before or during execution;
- binds model revision, policy, layer/dose plan, resolved positions, candidate tokens and
  source/target token tensors with deterministic SHA-256 evidence;
- omits prompts, token arrays, activations and candidate labels from the report.

## Example

The API accepts an already loaded model so production callers control model acquisition, device
placement and authentication:

```python
from structxai.core import Candidate
from structxai.patch_dose_response import (
    CrossSeedPolicy,
    DoseResponseAuditPolicy,
    PatchSweepSpec,
    require_cross_seed_dose_response_acceptance,
    require_dose_response_acceptance,
    run_residual_patch_sweep,
)

spec = PatchSweepSpec(
    model_revision="sha256:2df4...",
    layer_indices=(4, 8, 12),
    alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
    source_position=-1,
    target_position=-1,
    score_position=-1,
    positive_candidate=Candidate(" Ankara", (token_a,)),
    negative_candidate=Candidate(" Istanbul", (token_b,)),
)
report = run_residual_patch_sweep(
    model,
    source_input_ids=source_ids,
    target_input_ids=target_ids,
    spec=spec,
)
audit = require_dose_response_acceptance(
    report,
    policy=DoseResponseAuditPolicy(
        min_endpoint_recovery=0.20,
        min_monotonic_fraction=0.75,
        max_step_regression=0.05,
        max_abs_recovery=3.0,
        min_qualified_layers=2,
    ),
)

# Repeat the same governed experiment under independently selected seeds.
seed_audit = require_cross_seed_dose_response_acceptance(
    {11: report_seed_11, 22: report_seed_22, 33: report_seed_33},
    dose_policy=DoseResponseAuditPolicy(min_qualified_layers=2),
    cross_seed_policy=CrossSeedPolicy(
        min_seed_runs=3,
        min_seed_qualification_fraction=0.80,
        min_endpoint_sign_agreement=0.80,
        max_endpoint_range=0.25,
        max_auc_range=0.25,
        min_stable_layers=2,
    ),
)
```

`model_calls` is known before execution: one source capture, one target capture, every
layer-by-alpha intervention, and one final target repeat.

## Prespecified promotion gate

`audit_dose_response` evaluates every declared layer rather than returning only the best-looking
curve. Per layer, it records endpoint recovery, trapezoidal normalized AUC, the fraction of
non-regressing steps, the largest regression, the largest absolute recovery, stable finding codes,
and a qualification decision. The terminal audit is accepted only when the configured minimum
number of layers qualifies.

`require_dose_response_acceptance` is the fail-closed CI/promotion entry point. It raises
`DOSE_RESPONSE_AUDIT_REJECTED` for scientifically valid evidence that misses policy. Malformed,
non-canonical, internally inconsistent, or digest-tampered evidence raises a distinct
`PatchSweepError`. The audit binds the complete sweep digest, sweep-plan digest, policy digest and
all layer assessments into a new deterministic SHA-256 evidence digest.

The policy must be fixed before inspecting the sweep. A digest proves which policy was evaluated;
it does not prove that the policy was independently preregistered or that a rejected run was not
discarded.

## Cross-seed promotion gate

`audit_cross_seed_dose_response` first revalidates every child sweep and reruns the same
`DoseResponseAuditPolicy`; a supplied per-seed acceptance flag is never trusted. It then requires
the model revision, source and target input digests, sweep plan, execution policy, layer set and
alpha ladder to match exactly. Seed labels are non-negative integers and are sorted before hashing,
so mapping insertion order cannot change the result.

For every layer the aggregate records:

- the fraction of seeds passing the per-run dose-response policy;
- endpoint-recovery mean and range;
- normalized-AUC mean and range;
- the fraction of endpoints sharing the most common effect sign;
- stable finding codes and a cross-seed qualification decision.

The number of accepted seed runs and stable layers is bounded by `CrossSeedPolicy`. The terminal
evidence includes each seed label plus its sweep and per-run audit digests, both policy digests, all
aggregate metrics, and a deterministic SHA-256 digest. Raw prompts, tokens and activations are not
copied into the aggregate. `require_cross_seed_dose_response_acceptance` raises
`CROSS_SEED_DOSE_RESPONSE_REJECTED` for valid but unstable evidence; malformed, tampered or
mismatched runs fail separately.

## Authenticated seeded-run provenance

Caller-provided seed labels alone do not show that a runtime actually applied the seed. The
`PatchRunSigner` boundary seeds Python, NumPy's legacy global generator, CPU Torch, and every
available CUDA generator immediately before the sweep callback. It hashes the pre/post RNG states,
validates the returned sweep evidence, and restores the caller's original states even when
execution raises. A process-wide lock serializes this global-state boundary; callers can set a
bounded acquisition timeout and receive `RNG_BOUNDARY_BUSY` instead of racing another run. Raw RNG
state and secret key bytes are never copied into the receipt.

```python
from structxai.patch_run_receipt import (
    PatchRunSigner,
    audit_authenticated_cross_seed_dose_response,
)

signer = PatchRunSigner.from_env()
report, receipt, model_manifest, tokenizer_manifest = signer.run_from_artifacts(
    lambda: run_residual_patch_sweep(
        model,
        source_input_ids=source_ids,
        target_input_ids=target_ids,
        spec=spec,
    ),
    seed=11,
    model_path="artifacts/model",
    tokenizer_path="artifacts/tokenizer",
)

authenticated = audit_authenticated_cross_seed_dose_response(
    {11: report_11, 22: report_22, 33: report_33},
    {11: receipt_11, 22: receipt_22, 33: receipt_33},
    verification_keyring,
)
```

Configure signing material outside the repository:

```bash
export STRUCTXAI_PATCH_RECEIPT_KEY_ID=prod-2026-q4
export STRUCTXAI_PATCH_RECEIPT_KEY_B64="$(openssl rand -base64 32)"
```

Each manifest streams regular files into per-file SHA-256 commitments, sorts normalized relative
paths, and commits the file list plus byte counts to an aggregate digest. File-count, individual-file,
total-byte, and read-chunk limits are explicit. Empty trees, symbolic links, special files, and
artifacts that change during hashing or callback execution fail closed. Hugging Face cache snapshots
often contain symlinks; materialize a regular-file export with `save_pretrained` before using this
strict boundary.

Verification uses constant-time HMAC comparison, rejects unknown keys and stale or future-dated
receipts, and requires one receipt per seed. The authenticated aggregate also rejects mixed model
or tokenizer artifacts and mixed CUDA device counts, sorts runs before hashing, and binds receipt
digests to the existing cross-seed evidence digest. CUDA-less hosts commit a deterministic
zero-device state, so the same receipt schema is exercised in CPU-only CI. NumPy maps the governed
64-bit seed into the legacy generator's 32-bit seed space with modulo `2**32`.

## Interpretation and limits

A smooth or monotonic dose response is useful evidence that a reported patch effect is not merely a
single endpoint artifact. It does **not** prove that the patched state is on-manifold, that the site
is necessary or sufficient, that the mechanism is unique, or that the candidate decision is
semantically correct. Source and target positions remain a governed experimental choice. The
runtime accepts one unpadded sequence per side. The legacy signer path can still bind externally
computed digests; production workflows should prefer `run_from_artifacts`, which computes and
rechecks bounded manifests around the run. A manifest proves local bytes and paths, not their
publisher, training lineage, or semantic suitability.

Cross-seed agreement reduces dependence on one random run, but it does not establish statistical
independence. The signed boundary proves that the library set the covered process-global RNG states
at the callback boundary; it cannot prove that a model used those generators. Independent NumPy
`Generator` instances, distributed data-loader workers, remote inference servers, and third-party
RNGs remain outside the boundary. CUDA coverage is capability-aware but CPU-only CI simulates the
multi-device API rather than executing real kernels. Receipts are freshness checked but not
single-use. Larger uncertainty claims should use more than the default three-run minimum.

The next increment is propagating authenticated per-worker seeds into distributed data-loading and
recording the worker topology without exposing raw samples or RNG state.
