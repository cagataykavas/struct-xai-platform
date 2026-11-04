# Independent publication-reference custody

`structxai.stability_reference_custody` removes a single caller-supplied publication
reference from the final XAI evidence trust path. A bounded set of independent
custodians signs the exact canonical reference, the complete custodian set, and a
minimum approval threshold before the publication/runtime audit is accepted.

## Create the custody attestation

Each adapter implements the existing `ExternalEvidenceSigner` protocol. In production,
use separate KMS/HSM keys and administrative domains rather than multiple aliases for
one key:

```python
from structxai.stability_reference_custody import (
    create_publication_reference_custody_attestation,
)

custody_attestation = create_publication_reference_custody_attestation(
    independently_obtained_publication_reference,
    {
        "research-governance": governance_kms_signer,
        "release-operations": operations_hsm_signer,
        "external-auditor": auditor_signer,
    },
    required_approvals=2,
    attested_at_epoch=trusted_now_epoch,
    expires_at_epoch=trusted_now_epoch + 600,
)
```

The signer keys and immutable key-revision digests must be distinct. Custodians are
ordered canonically, and every signature binds:

- the exact publication-reference SHA-256;
- the digest of the complete ordered custodian set;
- the required approval count;
- the issuance and expiry epochs; and
- that approval's custodian ID, algorithm, key ID, key revision, and encoding.

This prevents an intermediary from lowering the threshold or swapping, adding, or
removing custodians while retaining valid signatures. The attestation itself is
self-digested so storage mutation is detectable.

## Verify one retained evidence unit

The deployment supplies a trusted public-key registry separately from the lineage-key
registry. The wrapper requires the custody quorum first, then performs the existing
publication-byte and signed runtime-lineage cross-binding:

```python
from structxai.stability_reference_custody import (
    serialize_custodied_stability_audit_receipt,
    verify_custodied_stability_audit_bundle,
)

receipt = verify_custodied_stability_audit_bundle(
    independently_obtained_publication_reference,
    custody_attestation,
    downloaded_publication_bytes,
    retrieved_uri=requested_uri,
    lineage_bundle=runtime_bound_signed_evidence,
    custody_verifiers=trusted_custody_public_keys,
    lineage_verifiers=trusted_lineage_public_keys,
    verification_epoch=trusted_now_epoch,
)
receipt_bytes = serialize_custodied_stability_audit_receipt(receipt)
```

The quorum tolerates unavailable or invalid approvals when enough independently trusted
approvals remain. Malformed schemas, duplicate custodian identities, reused key IDs or
key revisions, reference substitution, threshold changes, and insufficient valid keys
fail closed. Provider errors are converted to stable boundary errors without copying
private diagnostics.

Attestations must be issued at or after the publication epoch and expire within one
hour. Verification rejects expired attestations and attestations dated more than five
minutes in the future. The verifier requires an explicit epoch instead of reading the
host clock implicitly, so a protected runner can supply time from its approved source.
The signed window and applied verification time/skew are committed into the verification
digest; changing the window invalidates every custodian approval.

The final receipt contains only evidence digests and approval counts. It does not copy
the public URI, custodian/key identifiers, signatures, prompts, task IDs, records,
weights, or layer scores. Retain the original publication record, lineage envelope,
custody attestation, base audit receipt, and final receipt together.

## Trust boundary

This is multi-key authorization of a reference, not proof that the custodians are
organizationally independent. A deployment must provision keys under genuinely
separate control planes, distribute trusted public keys out of band, and define key
rotation and revocation policy. The module does not fetch records or keys, authenticate
the publication service, establish trusted time, authenticate the supplied verification
epoch, provide transparency-log consistency, prove immutable storage, or attest model
execution. Those controls remain with the protected runner and evidence-retention system.
