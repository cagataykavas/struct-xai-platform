# Path-attribution convergence audit

Integrated Gradients and related path methods approximate an integral with a finite number of model
evaluations. A visually plausible heatmap at 32 steps is not evidence that 32 steps were sufficient.
Changing the integration resolution can preserve the attribution sum while moving contribution among
features, or can leave a large completeness residual hidden by a convenient stopping point.

`structxai.path_convergence` audits a prespecified ladder of estimates before an explanation artifact is
accepted. It verifies the exact integration-step schedule and feature alignment, then evaluates:

- final completeness error relative to the observed model-output delta;
- relative L1 change between the final two attribution vectors;
- cosine agreement between those vectors;
- improvement of completeness residual from the first to final estimate;
- minimum output contrast and a governed failed-case fraction.

The artifact binds the benchmark, model weights and method configuration with canonical SHA-256
digests. Reports include only hashed case identities, bounded metrics, stable finding codes and
content-addressed artifact/policy evidence. Strict JSON parsing rejects duplicate keys, non-finite
values, unknown fields, stale artifacts and resource-budget overruns.

## Recommended workflow

1. Prespecify a doubling ladder such as 8, 16, 32 and 64 steps and thresholds before evaluation.
2. Hold the input, baseline, target output, model state and method configuration fixed.
3. Emit every attribution vector and the independently observed output delta.
4. Run the audit without selecting a more favorable stopping point after inspecting results.
5. Increase the maximum step count or reject the explanation when completeness or vector convergence
   fails.

## Limitations

Numerical convergence is necessary evidence for a path approximation, not proof of faithfulness,
causality or human usefulness. Two successive estimates can agree before the true integral is reached,
especially with non-smooth models. Completeness can also hold while attribution is distributed among
features incorrectly. Baseline choice, target definition and path construction remain separate
scientific decisions. SHA-256 binds evidence but does not authenticate the runner.

The next increment is to emit this artifact directly from a batched PyTorch Integrated Gradients runner
and calibrate step schedules by model family, input length and target-margin scale.
