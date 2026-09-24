# Activation-patching control gate

A large activation-patching effect is not automatically evidence for a localized mechanism. A
patch can move the output because almost any intervention at that site is disruptive, because the
corruption barely changed the behavior, or because unrelated patch locations produce comparable
effects.

This gate audits paired candidate-logit margins for each case:

- the clean and corrupted runs establish the recoverable behavioral gap;
- the hypothesized target patch measures directional recovery;
- a sham patch at a prespecified non-causal site measures procedural leakage; and
- multiple random-site patches establish a negative-control effect distribution.

The normalized target recovery is the target patch's movement toward the clean margin divided by
the clean–corrupted gap. Sham and random-control effects use absolute movement divided by the same
gap. Admission requires a material corruption effect, bounded target recovery, low sham/control
effects and explicit separation between target recovery and the random-control p95.
Every artifact also carries a bounded `experiment_id`, which is echoed in accepted reports so the
decision remains attributable when multiple experiment artifacts are evaluated in CI.

```bash
python -m structxai.patch_controls artifacts/patch-controls.json \
  --min-cases 20 \
  --min-random-controls 10 \
  --min-target-recovery 0.50 \
  --max-control-p95-effect 0.25
```

Exit code `0` accepts the artifact, `2` identifies malformed evidence or policy, and `3` rejects a
well-formed experiment. Reports contain margins and site identifiers but no prompts or activations.

## Experimental requirements and limits

Target, sham and random sites must be declared before examining outcomes and must use the same
model checkpoint, token position, corruption, candidate pair and scoring path. Random sites should
be sampled from a defensible eligible set; hand-picking weak controls invalidates specificity.

The gate does not prove that a component is necessary, sufficient, uniquely identifiable or part
of a human-readable circuit. Candidate first-token margins are a narrow behavioral endpoint.
Controls are paired diagnostics, not independent observations, and the empirical p95 is not a
calibrated p-value. Multi-token behavior, multiple-comparison correction, resampling uncertainty
and replication across prompts/seeds remain experiment-runner responsibilities.
