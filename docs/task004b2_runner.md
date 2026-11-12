# TASK-004B.2 — Full development cache runner (technical protocol)

Status: **CPU test suite passed in reference container; local Windows GPU batch pending.**
No full development inference was performed while preparing this patch.

## Scope / unchanged protocol

- Immutable full plan SHA-256: `39c14b72d7bd82b815757147838e45fe3df56bddf4616367139cfd5661bd089f`.
- Processed SPY SHA-256: `4375f3b90c5dbfe1cfd1a8191a36c15c2d878993ef04b130c8de8810927103ba`.
- `1258` development origin/target pairs; indices `3272..4529`, targets `3273..4530`.
- Input: 256 causal daily decimal log returns ending at the origin; output target at next trading session. The first context can include preceding warm-up observations.
- Frozen `amazon/chronos-t5-small` / `Salesforce/moirai-1.0-R-small` with the existing pinned revisions, hashes, and separate Python virtual environments.
- 100 samples; raw quantiles `.1/.5/.9`; seed `2027 + origin_index`; replay first 2 origins, tolerance `1e-5`; reused TASK-004A record schema.
- Scientific risk flags remain `false`, neither claim is cleared.
- No validation, final test, calibration, CQR or ACI.

## Additive files

- `scripts/full_development_cache.py` — orchestrating CLI and model-worker entrypoint. Imports TASK-004A runner contracts and TASK-004B.1 full plan module. Does not edit accepted source files.
- `tests/test_task004b2_runner.py` — CPU mock-adapter fault-injection tests.
- `docs/task004b2_runner.md` — this specification.

## One-command GPU batch on Windows

Run from `C:\AI-Workspace\projects\cita2027-phase2` with `.venv` already available. The master script automatically invokes `.venv/Scripts/python.exe` for Chronos and `.venv-moirai/Scripts/python.exe` for Moirai:

```powershell
& .\.venv\Scripts\python.exe .\scripts\full_development_cache.py pilot --limit 12 --batch-size 4
```

This executes and verifies the **first 12 origins of the full plan** in each environment (24 predictions total), not the old evenly-spaced TASK-004A pilot. It emits a read-only audit status and, only after both models pass, creates an immutable `small_batch_gate.json` with both sets of per-origin SHA-256 values. It does not write any completed-cache CSV/manifest for the partial batch. A failed model returns exit code 1 and does **not** issue a successful audit or gate. Re-running the identical command resumes valid origin records.

## Full run (ONLY after accepted GPU batch)

One PowerShell command for both backbones, automatic resume, final file-integrity and cross-model audit:

```powershell
& .\.venv\Scripts\python.exe .\scripts\full_development_cache.py run --allow-full --batch-size 100
```

The full runner refuses to run unless it finds and revalidates a small-batch gate bound to the identical full plan and first-N record hashes. The two model workers run **sequentially**, not concurrently. Per-origin JSON files are saved atomically, with a model-local exclusive lock. `--batch-size` controls progress/checkpoint cadence, not inference sample batch dimension.

## Optional commands

```powershell
& .\.venv\Scripts\python.exe .\scripts\full_development_cache.py preflight
& .\.venv\Scripts\python.exe .\scripts\full_development_cache.py audit --limit 12
& .\.venv\Scripts\python.exe .\scripts\full_development_cache.py audit
```

`audit --limit 12` checks partial-prefix records; `audit` without a limit demands both complete 1258-record manifests and cross-model alignment. A successful full audit returns `FULL_DEVELOPMENT_FILE_AUDIT_PASS_NOT_SCIENTIFIC_CLEARANCE`. Existing pilot artifacts under `pilot/` are never read as forecast records.

## Storage, atomicity, and recovery

```
artifacts/forecast_cache/full_development/
  plans/<plan_sha256>.json                      # Already published by TASK-004B.1
  runs/<plan_sha256>/
    small_batch_gate.json                       # Only after audited small GPU batch
    chronos/
      records/<origin_index_6digits>.json       # Atomic write-once
      predictions.csv                           # Created only at full completion
      manifest.json                             # Created only at full completion
    moirai/
      records/...
      predictions.csv
      manifest.json
    audit_report.json                           # Created only after complete cross-model audit
```

Resume verifies **every existing origin record** before loading the model and refuses noncanonical, malformed, incorrect-target or unexpected `.json` files. Each validated existing origin is skipped. Writer uses a temporary `.stage-*` file and atomic no-replace publication; temp artifacts in the records directory may be removed after the exclusive run lock is acquired. A crash before a record is published leaves the origin missing and eligible for safe recomputation. Partial batches never generate a misleading `manifest.json` or `predictions.csv`. A crash after publishing the full CSV but before manifest is supported if bytes match all records. Hashes of all individual records and the combined CSV are checked at completion.

**Lock recovery:** A lock file `.full.lock` indicates a possibly active writer. Do not delete it while any worker process is running. If the process was killed, verify that the recorded PID no longer exists, copy logs, and remove *only* the stale lock file (never the valid JSON records) before rerunning the same command. Do not edit corrupted published records in place; quarantine evidence and investigate. File-integrity validation intentionally refuses silent recomputation or overwrites.

## Resource expectation, not acceptance evidence

Previously audited 24-origin pilot GPU timings (excluding model load / replay): Chronos 1.307 s; Moirai 6.168 s. Simple linear scaling to 1258 origins suggests ~69 s for Chronos and ~323 s for Moirai **of inference only**, plus model load, record serialization/verification, GPU synchronization and audits. Peak PyTorch allocations 940.48 MiB and 75.25 MiB exclude model loading and do not represent total process VRAM. Disk preflight requires >=1 GiB free; output size must be measured on Windows rather than inferred from allocation numbers.

## Acceptance / controls

- **Batch GO:** 12/12 valid prefix records per model, matching full plan hash, checkpoint/data identities, origin alignment, hash/canonical integrity, causal context, ordered finite quantiles, replay checks, both risk flags false, and `SMALL_BATCH_GATE_ACCEPTED`.
- **Full GO:** only after *real Windows GPU batch logs* and a successful gate. Not granted by CPU mock tests alone.
- **Final full audit:** 1258 unique origins/model; no validation/final targets; exact original plan; validated record SHA/manifest/CSV; cross-model equality of causal origin/target identity; deterministic seeds. Does not imply strict out-of-training evaluation.
