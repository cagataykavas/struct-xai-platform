# Activation-site scan FWER audit

Activation patching sweeps often inspect many layer/token sites and then report the
largest recovery. Even when every site is null, the maximum grows with the number
of inspected sites. Treating that winner as if it were a single prespecified test
creates a multiple-comparison failure.

`structxai.patch_scan_fwer` is an offline release gate for a prespecified site
family. It uses jointly generated null replicates and a single-step max-statistic
to control the family-wise error rate (FWER) for each case.

## Statistical contract

For a clean candidate margin \(m_c\), corrupted margin \(m_x\), and patched margin
\(m_s\) at site \(s\), directional recovery is

\[
r_s = \frac{m_s - m_x}{m_c - m_x}.
\]

This definition works when corruption moves the margin in either direction. Cases
with a near-zero clean-to-corrupted gap are rejected because normalized recovery
would be unstable.

Every null replicate must contain the same complete eligible-site set. For null
replicate \(b\), the gate records the maximum recovery across that family:

\[
M_b = \max_s r_{b,s}.
\]

The one-sided, finite-sample corrected p-value is

\[
p_s = \frac{1 + \sum_b \mathbb{1}[M_b \ge r_s]}{B + 1}.
\]

A site passes only when its corrected p-value is at most `family_wise_alpha` and
its recovery is inside the configured effect-size range. The `+1` correction
prevents a zero p-value. Policy construction fails if the minimum permitted null
replicate count cannot resolve the configured alpha.

The gate also requires a minimum fraction of cases with at least one corrected
site and reports sites that recur across a configured fraction of cases. These
aggregate checks are deployment policy; they are not additional statistical
claims.

## Evidence contract

The JSON artifact binds the scan to a model, patch configuration, eligible-site
manifest, experiment identifier, and UTC creation time. Each case supplies clean
and corrupted margins, all observed site margins, and aligned joint-null
replicates. `eligible_sites_sha256` is the SHA-256 of the canonical JSON array of
sorted site IDs, so the declared family is checked against the actual scan:

```json
{
  "schema_version": 1,
  "experiment_id": "capital-scan-v3",
  "model_sha256": "<64 lowercase hex characters>",
  "patch_config_sha256": "<64 lowercase hex characters>",
  "eligible_sites_sha256": "<64 lowercase hex characters>",
  "created_at": "2026-09-27T00:59:30Z",
  "cases": [
    {
      "case_id": "case-001",
      "clean_margin": 1.0,
      "corrupted_margin": 0.0,
      "sites": [
        {"site_id": "layer.7.token.4", "patched_margin": 0.8}
      ],
      "null_replicates": [
        {
          "replicate_id": "permutation-001",
          "sites": [
            {"site_id": "layer.7.token.4", "patched_margin": 0.1}
          ]
        }
      ]
    }
  ]
}
```

Production artifacts need enough cases, eligible sites, and null replicates to
satisfy policy; the abbreviated object above only documents the shape.

Run the audit without loading a model:

```bash
python -m structxai.patch_scan_fwer artifacts/patch-scan.json \
  --output artifacts/patch-scan-audit.json
```

Exit codes are stable for automation:

| Code | Meaning |
| --- | --- |
| `0` | Artifact accepted |
| `2` | Well-formed artifact rejected by policy |
| `3` | Malformed artifact, I/O error, or report-write failure |

The report is written atomically. It contains only canonical SHA-256 identifiers,
bounded finding codes, normalized summary metrics, and policy/artifact digests;
raw case IDs, site IDs, and margins are not copied into the report.

## Fail-closed behavior

The parser rejects duplicate JSON keys, unknown fields, non-finite or extreme
numbers, duplicate identifiers, stale or future-dated evidence, mismatched site
families, inconsistent replicate counts, and resource-budget overruns. Collection
order does not affect the artifact digest.

## Required null generation

Null replicates must be produced *jointly* across every eligible site with the same
randomization or permutation draw. This preserves cross-site dependence for the
max statistic. Independently shuffling each site or omitting inconvenient sites
does not satisfy the contract. The eligible family and patch configuration must be
fixed before observed effects are inspected.

## Limitations

- FWER validity depends on an exchangeable, correctly generated joint null. The
  gate validates artifact structure, not the honesty of its producer.
- The single-step maximum statistic is intentionally conservative and does not
  provide effect confidence intervals or step-down adjusted p-values.
- Correction applies to the eligible sites within each case. It does not silently
  cover additional models, datasets, prompts, metrics, or analyses tried outside
  the bound artifact.
- A statistically unusual recovery is not proof of a human-interpretable causal
  mechanism, semantic correctness, or downstream safety.
- Recurrence across cases is a release heuristic. Correlated cases and dataset
  selection still require separate experimental review.

The next integration step is for the activation-patching sweep runner to emit the
bound artifact directly, including a versioned null-generation manifest and a
signed producer receipt.
