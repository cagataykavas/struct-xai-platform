# Token perturbation integrity audit

Deletion, masking and replacement curves are only meaningful when the generated model input differs exactly where the experiment says it differs. Character offsets can retokenize neighboring text, padding can be malformed, candidate tokens can move or disappear, and a nominal mask can silently become an unrelated replacement. A strong behavioral effect under any of those conditions is not valid faithfulness evidence.

`structxai.perturbation_integrity` independently reconciles a padded token trace with a declared splice contract. It verifies:

- contiguous right-padding and exact pad-token values;
- vocabulary bounds and binary attention masks;
- the declared deletion, replacement or one-mask-per-target operation;
- exact reconstruction of every active token outside the splice;
- preservation and relocation of the governed candidate span;
- exclusion of candidate and reserved special-token positions;
- no-op rejection;
- immutable model/tokenizer revisions, prompt digest and policy identity;
- evidence freshness and bounded input, sequence, variant, token and finding volumes.

Run the audit before interpreting a deletion or occlusion curve:

```bash
python -m structxai.perturbation_integrity \
  --input artifacts/token-perturbations.json \
  --output artifacts/token-perturbation-audit.json \
  --expected-model-id Qwen-0.5B \
  --expected-model-revision "$MODEL_COMMIT" \
  --expected-tokenizer-id Qwen-tokenizer \
  --expected-tokenizer-revision "$TOKENIZER_COMMIT" \
  --expected-policy-id perturbation-policy-v1 \
  --expected-prompt-sha256 "$PROMPT_SHA256"
```

Exit code `0` means accepted, `2` means structurally valid evidence violated policy, and `3` means the evidence was malformed or could not be read safely. Reports contain deterministic reason codes, aggregate counts, hashed experiment/variant references and canonical SHA-256 identities; they do not repeat input token IDs or replacement tokens.

The six `--expected-*` arguments form one execution lock and must be supplied together. A mismatch in the model ID, model revision, tokenizer ID, tokenizer revision, perturbation policy ID or prompt digest rejects the artifact with a stable `IDENTITY001`–`IDENTITY006` reason code. The report stores only the lock's canonical SHA-256 digest, so independent promotion gates can compare the configured contract without copying those identifiers into every report.

## Trust boundary

The artifact producer must capture tensors from the actual model call. The execution lock proves that the producer declared the expected identities, not that it used them, and SHA-256 supplies content identity rather than producer authenticity. The audit proves that the supplied token sequences obey their declared perturbation contract; it does not prove that the perturbation is in-distribution, semantically neutral, causally sufficient, fair, or representative. Production promotion should sign the report and combine it with random/inverse controls, behavioral faithfulness metrics and provenance-locked model execution.
