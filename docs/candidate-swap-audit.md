# Candidate-swap attribution audit

Pairwise explanations have a useful implementation invariant: when the same
scoring function is evaluated as `margin(A, B)` and `margin(B, A)`, the scalar
margin and its signed feature attributions should negate. Candidate labels may
swap position, but the aligned feature set should not change.

This audit turns that invariant into a bounded, model-free release gate. It is
intended to catch candidate-order leakage, reversed-label bugs, asymmetric
scoring wrappers, and attribution pipelines that silently align the two runs to
different features.

## Evidence contract

The input is JSON. Each case records both candidate orders and attribution
values aligned by stable feature ID:

```json
{
  "schema_version": 1,
  "cases": [
    {
      "case_id": "example-001",
      "candidate_a": "yes",
      "candidate_b": "no",
      "forward": {
        "candidate_order": ["yes", "no"],
        "margin": 1.2,
        "attributions": [
          {"feature_id": "prompt:0", "value": 0.7},
          {"feature_id": "prompt:1", "value": -0.2},
          {"feature_id": "prompt:2", "value": 0.1}
        ]
      },
      "swapped": {
        "candidate_order": ["no", "yes"],
        "margin": -1.2,
        "attributions": [
          {"feature_id": "prompt:0", "value": -0.7},
          {"feature_id": "prompt:1", "value": 0.2},
          {"feature_id": "prompt:2", "value": -0.1}
        ]
      }
    }
  ]
}
```

Case and feature identifiers must be unique and bounded. Candidate order must
be explicitly reversed, and the feature-ID sets must match exactly. Non-finite
numbers, zero attribution vectors, duplicate JSON fields, oversized inputs, and
incomplete evidence fail closed as invalid input rather than becoming a policy
failure.

## Metrics and decisions

For each case the gate computes:

- normalized margin antisymmetry error, `|m(A,B) + m(B,A)| / max(|m|)`;
- normalized attribution L1 residual, `sum(|a_i + a'_i|) / max(||a||_1)`;
- cosine similarity between the forward vector and the negated swapped vector;
- top-K absolute-salience overlap; and
- a minimum pairwise-margin check so near-zero decisions cannot appear to pass.

Threshold violations have stable reason codes. The report includes case-level
metrics, aggregate finding counts, the configured policy, and a canonical
SHA-256 digest of the validated evidence. It does not echo raw attribution
values. The release decision uses an explicit minimum case pass rate; the
default requires every case to pass.

Run it without loading a model:

```bash
python -m structxai.candidate_swap evidence/candidate-swaps.json \
  --output artifacts/candidate-swap-audit.json
```

Exit codes are suitable for CI:

- `0`: evidence accepted by policy;
- `2`: well-formed evidence rejected by policy;
- `3`: malformed evidence or invalid policy.

Thresholds should be calibrated on a prespecified, representative evaluation
set. The CLI exposes the principal thresholds while conservative resource
limits remain part of the Python policy object.

## Trust boundary and limitations

The producer is responsible for holding prompt text, tokenization, model state,
baseline, target scalar, attribution method, and feature mapping constant. The
gate verifies the supplied artifact, not the runner that created it. A signed
artifact or content-addressed storage is needed when producer integrity is in
scope.

Antisymmetry is only expected when the target is genuinely a signed pairwise
margin and the attribution method is applied consistently. It is not a valid
requirement for independently normalized probabilities, candidate-specific
prompts, order-sensitive multi-candidate objectives, or stochastic runs whose
randomness is uncontrolled.

Passing this audit demonstrates candidate-swap equivariance of the recorded
evidence. It does **not** establish attribution faithfulness, causal validity,
human usefulness, model robustness, or fairness. Perturbation tests,
parameter-randomization checks, negative controls, and domain review remain
separate evidence.
