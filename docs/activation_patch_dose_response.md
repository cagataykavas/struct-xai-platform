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

## Matched negative-control specificity gate

A smooth treatment curve can still be non-specific: unrelated source activations may produce the
same response. `structxai.patch_control_audit` compares the treatment with a prespecified set of
matched null-source sweeps before allowing a specificity claim:

```python
from structxai.patch_control_audit import (
    PatchControlPolicy,
    require_patch_control_specificity,
)

specificity = require_patch_control_specificity(
    treatment_sweep,
    {
        "unrelated-source-01": null_sweep_01,
        "unrelated-source-02": null_sweep_02,
        "corrupted-source-03": null_sweep_03,
    },
    dose_policy=DoseResponseAuditPolicy(
        min_endpoint_recovery=0.20,
        min_qualified_layers=2,
    ),
    control_policy=PatchControlPolicy(
        min_controls=3,
        min_endpoint_excess=0.15,
        min_auc_excess=0.10,
        max_control_endpoint_recovery=0.25,
        max_control_normalized_auc=0.20,
        min_specific_layers=2,
    ),
)
```

Every child sweep is fully revalidated and re-audited; supplied verdicts are never trusted. Each
control must use a distinct source-input commitment and exactly match the treatment's model
revision, target-input commitment, layer/alpha/candidate/position plan, execution policy, and layer
set. This prevents an easier target, different candidates, or a coarser dose ladder from being
presented as a null comparison.

For each layer, the gate compares the treatment endpoint and normalized AUC with the strongest
observed control. The treatment layer must already qualify under the ordinary dose policy, both
control ceilings must hold, and both treatment-minus-control margins must exceed their reviewed
minimums. The terminal policy requires a minimum number of specific layers. Evidence binds every
child sweep and rederived audit, hashes control labels, and retains only input/evidence commitments
and aggregate metrics—not prompts, tokens, activations, candidate labels, or raw control IDs.

Controls and thresholds must be selected before observing the treatment. This is a conservative
descriptive specificity check over a finite matched set, not a p-value or population-level null
distribution. Poorly chosen controls, shared confounders, or off-manifold control activations can
still mislead; acceptance does not establish necessity, sufficiency, uniqueness, or semantic truth.

## Prespecified task-cohort gate

A single task can still be selected because it produces a clean specificity result.
`structxai.patch_control_cohort` reruns the complete treatment/control audit for every task in a
fixed cohort and estimates the task-level specificity rate separately for every tested layer:

```python
from structxai.patch_control_cohort import (
    PatchControlCohortPolicy,
    PatchControlTask,
    require_patch_control_cohort,
)

cohort = {
    task_id: PatchControlTask(treatment=treatment, controls=matched_controls)
    for task_id, treatment, matched_controls in prespecified_tasks
}
cohort_audit = require_patch_control_cohort(
    cohort,
    cohort_policy=PatchControlCohortPolicy(
        min_tasks=20,
        family_confidence=0.95,
        min_specific_task_fraction_lower_bound=0.60,
        min_cohort_specific_layers=2,
    ),
)
```

Each layer receives a one-sided Wilson lower bound for the fraction of tasks that pass the matched
control gate. Bonferroni allocation across every layer gives simultaneous family-wise coverage
without assuming independence between layers. The cohort passes only when the reviewed number of
layers clears the reviewed lower-bound threshold. Exact task targets must be distinct, experiment
and policy identities must match, task IDs are hashed, and all child sweeps are revalidated rather
than trusting stored verdicts. The receipt binds the complete task inventory, child audits, policy,
method, hypothesis count, bounds, and descriptive mean excesses without retaining raw task or
control labels.

Wilson coverage is approximate and treats tasks as independent Bernoulli observations. Reused or
strongly related templates can make the bound optimistic; Bonferroni only addresses multiplicity
across layers. The task inventory and policy must be fixed before observing outcomes. This gate does
not prove that the cohort represents deployment traffic or that control interventions are on-manifold.

## Template-group bootstrap gate

Related prompts often come from one template, paraphrase family, language pair, or source dataset.
Counting those variants as independent tasks can make the task-level bound overconfident.
`structxai.patch_control_group_bootstrap` therefore requires a prespecified task-to-group inventory
and resamples whole groups rather than individual tasks:

```python
from structxai.patch_control_group_bootstrap import (
    PatchControlGroupBootstrapPolicy,
    require_patch_control_group_bootstrap,
)

grouped_audit = require_patch_control_group_bootstrap(
    cohort,
    task_groups={
        task_id: template_family_id
        for task_id, template_family_id in prespecified_task_groups
    },
    group_policy=PatchControlGroupBootstrapPolicy(
        min_groups=8,
        bootstrap_resamples=4_000,
        bootstrap_seed=20261108,
        family_confidence=0.95,
        min_group_specific_fraction_lower_bound=0.60,
        min_group_specific_layers=2,
    ),
)
```

For each layer, the audit first computes the specificity fraction within every group. It then
resamples groups with replacement and averages those group fractions, giving every reviewed family
equal weight regardless of how many closely related tasks it contains. A one-sided percentile lower
bound is computed with Bonferroni alpha allocation across layers. The same deterministic group draws
are reused across layers, and SHA-256 counter-mode sampling makes the receipt reproducible across
insertion orders without relying on a process-global RNG. Work, group count, and group size are
bounded before resampling.

The grouped receipt cross-binds the ordinary cohort audit, every rederived matched-control audit,
the complete task-to-group assignment, policy, method, work count, and layer results. Task and group
labels are domain-separated hashes; prompts, controls, tokens, activations, and raw identifiers are
not retained. A task-level rejection remains a rejection even if the grouped estimate passes.

The percentile bootstrap is approximate and assumes groups—not tasks—are exchangeable and
independent. Incorrect or post-hoc grouping, too few families, dataset overlap across families, and
nonrepresentative task selection can still produce misleading bounds. This gate does not establish
mechanism uniqueness, necessity, sufficiency, semantic correctness, or deployment validity.

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
