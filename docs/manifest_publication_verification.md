# Manifest publication verification

`structxai.stability_publication` verifies the exact bytes downloaded from the
publication URI before a manifest-locked explanation-stability cohort is executed.
This closes the gap between retaining a publication-reference digest and actually
checking the external record named by that reference.

## Publication flow

Create the canonical record before evaluation, publish those exact bytes, and retain
their SHA-256 in the lineage reference:

```python
import hashlib

from structxai.stability_lineage import create_manifest_publication_reference
from structxai.stability_publication import create_manifest_publication_record

record_bytes = create_manifest_publication_record(
    manifest,
    published_at_epoch=published_at_epoch,
)
publication_record_sha256 = hashlib.sha256(record_bytes).hexdigest()

# Upload record_bytes without reformatting or recompressing them.
reference = create_manifest_publication_reference(
    publication_uri=immutable_publication_uri,
    publication_record_sha256=publication_record_sha256,
    manifest_sha256=manifest["manifest_sha256"],
    published_at_epoch=published_at_epoch,
)
```

The record is printable canonical JSON containing the complete privacy-reduced task
manifest, its digest, and the claimed publication epoch. It contains manifest
commitments—not prompts, dataset rows, model weights, layer scores, or task reports.

## Protected-runner intake

The network client remains outside this package so the deployment can enforce its own
DNS, egress, certificate, redirect, proxy, and authentication policy. Pass the final
response bytes and the exact requested URI into the offline verifier:

```python
from structxai.stability_publication import verify_manifest_publication_record

publication_verification = verify_manifest_publication_record(
    reference,
    downloaded_response_bytes,
    retrieved_uri=requested_uri,
)

# Only after this succeeds:
task_reports, runtime_evidence = run_manifest_locked_stability_cohort(...)
```

Verification fails closed when:

- the retrieved URI differs from the signed-lineage reference;
- the raw response SHA-256 differs from the referenced record digest;
- the response is empty or larger than 512 KiB;
- JSON is non-ASCII, noncanonical, malformed, contains duplicate keys or non-finite
  constants, or has missing/extra fields;
- the record schema, manifest digest, or publication epoch differs from the reference;
  or
- the nested manifest fails its complete existing schema, policy, task-order, and
  self-digest validation.

The returned verification receipt retains aggregate byte count and commitments only.
It hashes the public URI rather than copying it and does not duplicate the nested
manifest.

## Trust boundary

This verifies downloaded content against a caller-trusted reference. It does not
perform the download, authenticate the publisher, establish trusted time, prove the
record existed before evaluation, detect publisher equivocation, or prove append-only
retention. A protected workflow should obtain the reference independently, prohibit
redirects or bind the final URL explicitly, restrict egress, retain the raw record and
verification receipt together, and use a transparency log or trusted timestamp when
prespecification time matters.
