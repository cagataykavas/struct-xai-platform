# Stability audit bundle

`structxai.stability_audit_bundle` closes the retained-evidence mixing gap between
manifest publication verification and signed runtime lineage verification. Both
components can be valid independently while naming different publication references.
The bundle verifier requires them to describe one exact XAI cohort before producing a
retention receipt.

## Verification flow

```python
from structxai.stability_audit_bundle import (
    serialize_stability_audit_receipt,
    verify_stability_audit_bundle,
)

receipt = verify_stability_audit_bundle(
    independently_obtained_publication_reference,
    downloaded_publication_bytes,
    retrieved_uri=requested_uri,
    lineage_bundle=runtime_bound_signed_evidence,
    verifiers=trusted_public_verifiers,
)
receipt_bytes = serialize_stability_audit_receipt(receipt)
```

The verifier first runs the complete existing publication-record validation and then
the complete signed runtime-lineage validation. It rejects the result unless both
components agree on:

- manifest SHA-256;
- publication-reference SHA-256;
- exact publication-record SHA-256; and
- claimed publication epoch.

This specifically prevents retaining an independently valid publication record beside
an independently valid runtime envelope from a different publication context.

## Retention receipt

The canonical `struct-xai-stability-audit-receipt/v1` document commits to the verified
publication, runtime inputs, report set, accepted cohort evidence, signed evidence,
signature-key revision, and both component verification results. Its own
`receipt_sha256` makes post-verification mutation detectable.

The receipt is privacy-reduced. It stores task count, signature algorithm, and digests;
cohort and key identifiers are hashed. It does not retain the publication URI, prompts,
dataset rows, model weights, task IDs, layer scores, task reports, or signatures. Retain
the raw publication record and signed lineage envelope separately because their digests
are the evidence needed to reproduce the receipt.

## Trust boundary

The verifier authenticates the lineage only through the caller-supplied trusted public
verifiers and treats the independently supplied publication reference as trusted input.
It does not download content, establish trusted time, authenticate the publication
service, prove that storage is immutable, detect equivocation across isolated auditors,
or attest the model host. Provider-specific download and retention controls remain
deployment responsibilities. A production workflow should use restricted egress,
independent reference custody, immutable versioned storage, and a separately controlled
verification key registry.
