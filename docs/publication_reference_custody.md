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
    key_status_snapshot=signed_custodian_key_status,
    key_status_authority_verifiers=trusted_key_status_authorities,
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

## Enforce custodian key revocation

`create_custody_key_status_snapshot` produces a separately signed, short-lived view of the exact
custodian key set. Every row binds a key ID and immutable key-revision digest to either `active` or
`revoked`. The snapshot must cover the attestation's complete key set exactly, expire within one
hour, and satisfy an externally configured authority quorum whose key revisions are distinct from
one another and from every custodian revision.

```python
from structxai.stability_reference_custody import create_custody_key_status_snapshot

signed_custodian_key_status = create_custody_key_status_snapshot(
    {
        governance_verifier.key_id: (governance_verifier.key_revision_sha256, "active"),
        operations_verifier.key_id: (operations_verifier.key_revision_sha256, "revoked"),
        auditor_verifier.key_id: (auditor_verifier.key_revision_sha256, "active"),
    },
    protected_key_status_authority_signers,
    required_authority_signatures=2,
    signing_authority_ids=(
        "key-status-governance",
        "key-status-security",
    ),
    issued_at_epoch=trusted_now_epoch,
    expires_at_epoch=trusted_now_epoch + 600,
)
```

The authority-policy digest commits the exact trusted authority key set and required threshold.
Each signature covers that policy, the complete status view, and its validity window. Verification
rejects threshold or authority-set substitution, reused authority revisions, missing quorum,
unknown signers, and malformed or invalid signatures.

When both key-status arguments are supplied to `verify_custodied_stability_audit_bundle`, revoked
revisions are removed before custody signatures are counted. The status snapshot, authority policy,
threshold, valid-authority count and status verification are committed into the custody verification
digest, which the final audit receipt already cross-binds. Omitting one of the two status inputs
fails closed. The legacy no-status path remains available for retained evidence, but a protected
production release should always supply both inputs from separately controlled configuration.

The final receipt contains only evidence digests and approval counts. It does not copy
the public URI, custodian/key identifiers, signatures, prompts, task IDs, records,
weights, or layer scores. Retain the original publication record, lineage envelope,
custody attestation, base audit receipt, and final receipt together.

## Retain an append-only status history

`structxai.stability_status_transparency` records a status snapshot only after the existing
authority-quorum verification succeeds. The JSONL log stores the snapshot and verification
commitments, observation epoch, sequence, and previous-entry digest; it does not copy key IDs,
key revisions, status rows, authority identities, or signatures.

```python
from structxai.stability_status_transparency import (
    append_verified_custody_key_status_snapshot,
    verify_custody_status_transparency_log,
)

checkpoint, appended = append_verified_custody_key_status_snapshot(
    "/var/lib/struct-xai/custody-status.jsonl",
    signed_custodian_key_status,
    expected_custodian_key_revisions,
    trusted_key_status_authorities,
    log_id="production-custody-status",
    required_authority_signatures=2,
    verification_epoch=trusted_now_epoch,
)

# Retain this checkpoint independently from the mutable log, then require the
# next observed log to extend it.
current_checkpoint = verify_custody_status_transparency_log(
    "/var/lib/struct-xai/custody-status.jsonl",
    log_id="production-custody-status",
    prior_checkpoint=previous_independently_retained_checkpoint,
)
```

Append and verification take advisory file locks. Exact snapshot replay is idempotent, while
concurrent writers serialize before assigning a sequence. Every append is flushed with `fsync`;
the complete bounded history is revalidated before use. The log must be a regular file with mode
`0600`, and the Linux implementation refuses a symlink at the leaf path. Run it beneath a
deployment-owned directory that untrusted users cannot replace, and publish each returned
checkpoint to a separately controlled, immutable retention system before authorizing the next
release.

This is a linear hash-chain consistency check, not a globally witnessed Merkle transparency
service. Verification is O(n), with explicit 16 MiB and 100,000-entry limits. A host administrator
who can replace both the log and every retained checkpoint can still rewrite history; an unsigned
checkpoint is trustworthy only because it was independently retained. The module also does not
establish trusted time or authenticate the protected runner that supplied the observation epoch.

## Trust boundary

This is multi-key authorization of a reference, not proof that the custodians are
organizationally independent. A deployment must provision keys under genuinely
separate control planes, distribute trusted public keys out of band, and define key
rotation and revocation policy. The module does not fetch records or keys, authenticate
the publication service, establish trusted time, authenticate the supplied verification epoch,
prove that a key-status authority quorum cannot collude, provide global transparency witnesses,
prove immutable storage, or attest model execution. Those controls remain with the protected
runner and evidence-retention system.
