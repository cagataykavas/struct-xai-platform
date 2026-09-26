# Attribution seed-stability audit

Approximate attribution methods can change their feature ranking, signs, or direction across random seeds even when the model output is fixed. A single attractive heatmap hides that uncertainty. This audit turns repeated attribution runs into bounded release evidence.

## Evidence contract

The JSON artifact uses schema `struct-xai-attribution-seed-stability/v1` and binds a benchmark to:

- a content-addressed model;
- an attribution method and content-addressed method configuration;
- timezone-aware evidence generation time;
- unique cases, unique random seeds, a fixed feature-ID set, finite attributions, and the model output margin for every run.

The default policy requires five runs per case. It measures every seed pair rather than comparing only to a convenient reference seed:

- signed cosine similarity over the complete attribution vector;
- Jaccard overlap between absolute-magnitude Top-K feature sets;
- sign agreement on the union of each pair's Top-K features;
- concentration of Top-K selections into one consensus set;
- model-output margin span, which detects confounding inference nondeterminism.

Any malformed, stale, future-dated, duplicate, non-finite, zero-vector, misaligned, or over-budget evidence is rejected before a policy decision. Case, run, feature, byte, and pair-comparison budgets bound CPU and memory work.

## CLI

```bash
python -m structxai.seed_stability_audit evidence.json --output report.json
```

Exit codes are stable: `0` means accepted, `2` means well-formed evidence failed policy, and `3` means malformed or over-budget evidence. Report replacement is atomic. Reports hash case, benchmark, and method identifiers and never copy prompts, tokens, or attribution values.

Thresholds must be fixed before evaluating a release and calibrated by method, model family, feature granularity, and task. The default zero-failure policy is intentionally conservative; a governed case-failure fraction can be configured for larger representative suites.

## Trust boundary and limitations

The audit trusts the producer to execute the declared model and method configuration and to vary only the attribution seed. SHA-256 identities provide binding, not authenticity; signatures or an attested runner are required against a malicious producer.

Seed stability is repeatability, not faithfulness. A consistently wrong explanation can pass. High agreement does not establish causality, completeness, fairness, calibration, robustness to input changes, or usefulness to a human. Conversely, an unbiased high-variance estimator can fail despite converging with more samples. The audit does not estimate confidence intervals or select a sufficient sample count.

## Next integration step

Emit this contract directly from the experiment runner, preserve it beside the model and method-config manifests, and calibrate seed count and thresholds with prespecified multi-seed studies. A later extension should report uncertainty intervals and convergence curves without choosing the stopping point on the evaluated evidence.
