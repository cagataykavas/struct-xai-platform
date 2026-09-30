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
from structxai.patch_dose_response import PatchSweepSpec, run_residual_patch_sweep

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
```

`model_calls` is known before execution: one source capture, one target capture, every
layer-by-alpha intervention, and one final target repeat.

## Interpretation and limits

A smooth or monotonic dose response is useful evidence that a reported patch effect is not merely a
single endpoint artifact. It does **not** prove that the patched state is on-manifold, that the site
is necessary or sufficient, that the mechanism is unique, or that the candidate decision is
semantically correct. Source and target positions remain a governed experimental choice. The
runtime accepts one unpadded sequence per side and binds a caller-supplied model revision, but does
not hash or authenticate model weights.

The next increment is a Hugging Face adapter that records tokenizer offset mappings, model-weight
provenance and signed run receipts, followed by prespecified monotonicity and cross-seed policies.
