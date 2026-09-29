# Tokenizer-migration attribution audit

Token-level explanations are indexed by a tokenizer's feature space. A
tokenizer revision can split Turkish suffixes, whitespace, punctuation, or
Unicode text differently even when the visible prompt and explanation target
are unchanged. Comparing token arrays by position can therefore hide a real
explanation change or report a false one.

`structxai.tokenizer_migration` provides a model-independent, fail-closed gate
for this migration boundary. It projects each content token's signed
attribution uniformly onto the token's UTF-8 byte interval, then compares the
two runs in that shared coordinate system.

## Evidence contract

```json
{
  "schema_version": "struct-xai/tokenizer-migration/v1",
  "created_at": "2026-09-29T00:59:50Z",
  "benchmark_digest": "1111111111111111111111111111111111111111111111111111111111111111",
  "model_weights_digest": "2222222222222222222222222222222222222222222222222222222222222222",
  "target_digest": "3333333333333333333333333333333333333333333333333333333333333333",
  "cases": [
    {
      "case_id": "tr-morphology-001",
      "input_digest": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
      "input_byte_length": 4,
      "reference": {
        "tokenizer_digest": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "output_margin": 1.25,
        "tokens": [
          {"start_byte": 0, "end_byte": 2, "attribution": 2.0},
          {"start_byte": 2, "end_byte": 4, "attribution": 2.0}
        ]
      },
      "candidate": {
        "tokenizer_digest": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "output_margin": 1.25,
        "tokens": [
          {"start_byte": 0, "end_byte": 4, "attribution": 4.0}
        ]
      }
    }
  ]
}
```

The artifact contains no prompt text or token strings. The producer records a
content digest, the UTF-8 byte length, and positive half-open byte offsets.
Each run must cover `[0, input_byte_length)` exactly once, with no gap,
overlap, zero-width token, or special-token placeholder. Special-token
attributions must be handled separately by the producer rather than assigned
ambiguous `(0, 0)` offsets.

The benchmark, model weights, explanation target, input, and both tokenizer
configurations are content-addressed. A digest binds the supplied identity but
does not authenticate its producer.

## Decision metrics

For a token spanning `n` bytes, each byte receives `token_attribution / n`.
This conserves the signed attribution sum while producing aligned vectors.
The gate then evaluates:

- signed cosine similarity;
- relative L1 difference, normalized by the larger vector L1 norm;
- Top-K absolute-salience Jaccard overlap;
- model output-margin drift; and
- total signed-attribution drift.

Degenerate near-zero attribution vectors are rejected rather than producing an
unstable cosine. A policy may allow a prespecified fraction of failed cases,
but every case finding remains in the report. Artifact freshness violations
always reject the release.

```bash
python -m structxai.tokenizer_migration evidence.json \
  --output tokenizer-migration-report.json
```

| Exit | Meaning |
| ---: | --- |
| `0` | Well-formed evidence satisfies the governed aggregate policy |
| `2` | Well-formed evidence violates the release policy |
| `3` | Evidence, policy, or output is malformed/unavailable |

Reports contain bounded reason codes, aggregate metrics, and hashed case
references. They do not echo prompts, tokens, input digests, or attribution
vectors. Output replacement is atomic and created with owner-only permissions.

## Integration guidance

Generate both explanation runs from one immutable case manifest. Hold the
model weights, target, attribution algorithm, baseline, inference settings,
and normalization fixed; vary only the tokenizer revision and the compatible
input encoding path. Prespecify thresholds on a calibration corpus before
evaluating a release candidate.

The audit belongs before tokenizer promotion:

```text
reference runner + candidate runner
               |
        offset-bound artifact
               |
     tokenizer migration audit
          |             |
        accept        reject
          |             |
       promote      investigate spans
```

## Limits

- Uniform distribution inside a token is an explicit approximation. A token
  attribution does not identify which character or byte caused the score.
- Byte-level Top-K can split a multi-byte Unicode character. It is a stable
  alignment substrate, not a linguistic explanation unit.
- High agreement demonstrates migration consistency, not faithfulness,
  causality, correctness, fairness, or human usefulness. Two identically wrong
  explanations can pass.
- The gate trusts the runner to use the declared model, target, method, and
  input. SHA-256 is not a signature.
- A tokenizer change that intentionally changes model semantics may correctly
  fail this invariant and require a new model/explainer validation baseline.

The next step is direct artifact emission from the Hugging Face runner using
offset mappings, plus calibration on Turkish morphology and Unicode stress
slices with signed producer receipts.
