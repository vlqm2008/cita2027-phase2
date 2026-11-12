# TASK-011 Phase-A identity schema hotfix

The Windows run failed at `accepted_anchors` with
`PINNED_MODEL_IDENTITY_DRIFT:chronos`. The frozen local checkpoint probes had
already matched both checkpoint IDs, commits, and weight SHA-256.

**Root cause:** The accepted TASK-009 BTC evidence index records the key
`checkpoint_id` but the new TASK-011 `CHECKPOINTS` dictionary uses the key
`id`. Direct dictionary equality is invalid even when all three underlying
identities match. Both model records in the accepted manifest use
`checkpoint_id`.

**Change:** `scripts/task011_fix_checkpoint_schema.py` verifies the exact
unmodified TASK-011 `lock.py` source SHA before altering only its identity
comparison. The new comparison validates exact keys and the unchanged pinned
ID, revision, and weight SHA-256 for *both* models. It will reject unrelated
edits, schema changes, or a pre-existing finalized pre-test manifest. It is
idempotent on the known fixed SHA. It does not modify adapters, calibration,
Development/Validation/Final Test data, or checkpoints.

Focused regression tests: `tests/test_task011_identity_schema_hotfix.py`.

**Scientific gate:** This fix removes one software bug, not any other preflight
requirement. If a later Phase-A gate fails (e.g. source coverage, exact XNYS
session list, Binance checksums, source-vintage proof), stop and preserve the
log. No Phase-B inference should occur until the pre-test manifest is
published and SHA-verified. Provenance flags remain false.
