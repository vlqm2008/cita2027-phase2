# TASK-004A — causal frozen TSFM forecast-cache pilot (design v1)

**Scope**: Development-only, exactly 24 origins by default (configurable 20–50). Reuse
`configs/smoke.yaml`, unchanged adapters, `load_processed`, `make_window`, snapshot
manifest and checkpoint audit. No calibration, no evaluation, no full cache.

## Design invariants

- Row `i` is an **exchange-session observation**; its `log_return` is the daily decimal
  log return ending at session `i`. Forecast origin `i` is after that observation is
  available. The only predictor input is 256 past returns with row indices
  `[i-255, ..., i]`; label `log_return[i+1]` is persisted *after* inference but
  never passed to the backbone. No assumption about point-in-time availability of
  historically adjusted vendor prices is made.
- Both origin and target rows must have `split == development`. Target is the next
  row `i+1`, verified through `previous_row_index` and `previous_date`.
- Origin selector is independent of outcomes, using chronological development
  membership and equally spaced valid row positions. Pilot origins are hashed in
  a shared model-independent immutable origin plan. Every model must use exactly
  the same plan; never tune based on pilot labels.
- Protocol is inherited from the accepted smoke configuration: context 256,
  horizon 1, samples 100, quantiles (0.1, 0.5, 0.9) computed from sample paths
  via NumPy linear empirical quantiles, seed 2027, per-origin seed 2027 + row index.
  First 2 origins (smoke `replay_origins`) are repeated with original tolerance
  `1e-5`. **Only 2 origins are replay-checked, not all 24.**
- Model weights are frozen in the existing adapters; each model runs in a separate
  virtual environment. Local model weights are SHA-256 checked against pinned Hub
  LFS hashes before any forecast, including resuming.
- No claims of out-of-training validity, complete conditional coverage, or
  point-in-time adjusted-price validation.

## Minimal additive integration

This patch **only adds** these files to the existing repository:

```text
src/cita2027/forecasting/cache.py
scripts/pilot_forecast_cache.py
tests/test_task004a_cache.py
docs/task004a_pilot.md
```

No existing accepted file should be overwritten; do not copy the entire
Pre-Phase-3 reference archive onto your Windows repository.

## Cache layout and data contracts

```text
artifacts/forecast_cache/pilot/
  plans/<plan_sha256>.json
  runs/<plan_sha256>/
    chronos/
      records/<origin_index:06d>.json
      predictions.csv
      manifest.json
    moirai/
      records/<origin_index:06d>.json
      predictions.csv
      manifest.json
    audit_report.json
```

`plan_sha256` hashes canonical, sorted, indent-2 JSON with UTF-8 and trailing
newline. The plan lists origin/target indices and dates and context SHA values
but does **not** contain realized targets or predictions. All 24 origins are
selected once and reused by both models. The manifest records source hashes,
actual checkpoint revision and weight SHA-256, counts, exact protocol, python and
package versions, runtime excluding model loading and replay, GPU allocation
excluding loading, per-origin JSON SHA-256 and final CSV SHA-256.

Each per-origin record has `forecast-cache-pilot-v1` schema and fields:

- `schema_version`, `plan_sha256`, `model`, `checkpoint_id`, `checkpoint_commit`,
  `checkpoint_weight_sha256`, `processed_snapshot_id`, `processed_sha256`.
- `origin_index`, `target_index`, `origin_date`, `target_date`,
  `context_first_index`, `context_last_index`, `context_sha256`.
- Raw (`not calibrated`) `q10`, `q50`, `q90`, `target_log_return`.
- `seed`, `context_length`, `horizon`, `samples`, `quantile_levels`, `target_units`.
- `elapsed_seconds`, `peak_allocated_mib`, `replay_checked`,
  `replay_max_abs_diff`.
- `contamination_free_verified=false`,
  `point_in_time_adjusted_prices_verified=false`.

Two data formats are intentionally retained: per-origin JSON for resumability,
plus a CSV export for downstream readers. There is no new Parquet dependency.
`predictions.csv` is compiled only when all records exist.

### Atomic commits, resume, duplicate checks

- A per-run `.pilot.lock` is acquired with an exclusive create. **Do not run two
  processes for one model/plan concurrently.** A stale lock after a hard kill
  requires manually confirming the process is no longer alive before deleting it.
- Each record is written via fsynced temporary file and an atomic, no-overwrite
  same-volume hardlink; incompatible existing records or extra JSON files fail
  loudly. This assumes a local filesystem that supports hardlinks (normally NTFS).
- Re-launching the same command verifies hashes, checks previous records against
  the pinned data and plan, skips completed origins, and forecasts only missing
  origins. A completed run verifies CSV/record/manifest integrity and performs
  **zero** new inference. A final CSV orphaned by a crash can be recovered only
  when all per-origin records are complete and its bytes exactly match their
  canonical CSV export; otherwise the runner stops without overwriting.
- Completed output is immutable by policy; any corruption causes a hard failure.
- An audit after both backbones finish verifies their run manifests, every
  record, source/weight identities, CSV hashes, pairwise origin/target/context
  hashes and observed target equality, then publishes `audit_report.json` once.

## Exact Windows PowerShell procedure

From the existing repository root:

```powershell
$C = '.\.venv\Scripts\python.exe'
$M = '.\.venv-moirai\Scripts\python.exe'
$env:PYTHONPATH = (Resolve-Path .\src).Path

# CPU regression, run with the preserved Chronos environment:
& $C -m pytest -q

# The same pilot count and untouched smoke protocol for each model:
& $C .\scripts\pilot_forecast_cache.py --mode run --model chronos --device cuda --count 24 --root . --config .\configs\smoke.yaml --checkpoint-audit .\artifacts\checkpoints\hf_audit.json
& $M .\scripts\pilot_forecast_cache.py --mode run --model moirai --device cuda --count 24 --root . --config .\configs\smoke.yaml --checkpoint-audit .\artifacts\checkpoints\hf_audit.json

# CPU-only cross-backbone artifact integrity check; requires both completed:
& $C .\scripts\pilot_forecast_cache.py --mode audit --count 24 --root . --config .\configs\smoke.yaml --checkpoint-audit .\artifacts\checkpoints\hf_audit.json
```

Avoid re-downloading SPY, changing the checkpoint revisions, running validation or
final-test origins, or invoking pip in the Chronos environment. The pilot runner
will fail if the expected local checkpoint files cannot be verified offline.

## Acceptance gates

1. Before pilot: previous Phase 2/Pre-Phase-3 audits still PASS; exact checkpoint
   identities and full source hashes unchanged; no new dependencies in `.venv`.
2. Tests: contract tests pass; demonstrate future-target perturbation invariance,
   quantile ordering, duplicate detection, interruption/resume and cross-model
   alignment. Note a passing mock integration test is **not** GPU inference.
3. Runtime: both 24-origin runs produce fresh `PILOT_COMPLETE_UNAUDITED` manifests;
   24 records each, deterministic replay for first 2, CPU/GPU telemetry recorded,
   origins only from development. A 24-origin run does not imply a full cache.
4. Independent file audit prints
   `PILOT_FILE_AUDIT_PASS_NOT_SCIENTIFIC_CLEARANCE`; user reviews manifest, raw
   origin list, logs, timings and provenance flags. Only then mark TASK-004A
   pilot ACCEPTED and decide TASK-004B GO.
5. Remain **NO-GO** on any SHA mismatch, duplicate, shift in plan/checkpoint/seed,
   target alignment fault, replay failure, nonfinite/unsorted quantiles,
   incomplete manifest, missing GPU execution evidence, or accidental validation/
   final-test inclusion.

## Known research risks and limits

- Both contamination and point-in-time adjusted-price verification flags are
  explicitly `false`. Commit/release date does not establish an untouched test.
- `uni2ts==1.2.0` is used with PyTorch 2.9.1+cu128 outside upstream's declared
  `torch<2.5` constraint. This is an audited experimental compatibility exception.
- GPU peak allocated excludes model load and other device allocations; runtime
  excludes cold checkpoint load and deterministic replay. Do not extrapolate
  to full-cache costs from 24 points without measuring.
- Reproducibility is asserted only for same machine/environment and the tested
  replay subset. Bitwise agreement across different CUDA kernels, drivers or
  checkpoint-library releases is not guaranteed.
- Upstream `make_window()` returns the realized label as a separate field but
  only `window.context` is passed to the adapter. Review and test future code
  changes to preserve this information-set boundary.
