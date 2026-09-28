# Complete candidate sequence scoring

Struct-XAI's layer trace deliberately uses the first token of each candidate. That makes a hidden-state
projection inspectable at every layer, but it can disagree with the model's preference after the complete
candidate suffix is considered. A strong first token can be followed by an implausible continuation.

`run_layerwise` now attaches `final_candidate_sequence_scoring` to its analysis result. For each candidate,
the scorer appends the already-tokenized candidate suffix to the prompt and uses causal-LM teacher forcing to
collect the conditional log probability of every candidate token. It reports:

- total sequence log probability, the proper joint score for the supplied token sequence;
- mean token log probability, a length-normalized diagnostic;
- first-token and minimum-token log probabilities;
- deterministic rankings for all three decision rules;
- explicit flags when the first-token, total, and normalized winners disagree.

Candidates are padded and scored in one bounded batch, so the check adds one model forward pass per prompt
rather than one pass per candidate.

The extra evidence makes proxy failures reviewable without pretending that output probability is itself a
causal explanation. Layer-wise projections and output-level sequence scoring answer different questions and
remain separately named in the artifact.

## Guardrails

- A single unpadded prompt is required, so the causal boundary cannot silently move.
- Candidate labels must be non-empty and unique.
- Candidate count and suffix length are bounded (64 candidates and 128 tokens by default).
- Invalid token ids, vocabulary overflow, misaligned model logits, and non-finite scores fail closed.
- Ranking ties are resolved by label for deterministic artifacts.
- No per-token probability vectors are persisted; the report is bounded to aggregate evidence.

## Tokenization contract

Candidate suffixes use the same `tokenizer.encode(label, add_special_tokens=False)` contract as the existing
first-token metric. This is intentionally recorded as
`candidate_suffix_tokenized_separately_without_special_tokens`. Callers should include the whitespace or
other boundary marker required by their model in the candidate label, as the existing examples do.

Total log probability naturally favors shorter candidates. Mean log probability is included to expose that
sensitivity, but it is not a sequence probability and must not be substituted silently. Disagreement between
the total and mean winners is evidence that the downstream decision rule needs to be chosen explicitly.

## Methodological boundary

Teacher-forced likelihood does not establish that a layer, token, or neuron caused the answer. It also does
not calibrate probabilities across models, and it inherits the supplied tokenizer/model revision. The field is
an output-level validity check for the first-token proxy; causal claims still require controlled interventions.
