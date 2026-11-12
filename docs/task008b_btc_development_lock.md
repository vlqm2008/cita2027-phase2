# TASK-008B — BTC Development frozen inference and calibration

## Scope and immutable inputs

This is an **additive patch** for existing `cita2027-phase2`. It does not alter accepted SPY files, the GPU adapters, `cita2027.calibration.online`, the BTC ingestion pipeline, or any frozen model checkpoint. No network downloading is performed by the new runner except potentially library internals; checkpoint verification requires an existing local file with `local_files_only=True`.

Hard-pinned accepted BTC artifacts:

| Artifact | SHA-256 |
|---|---|
| BTC daily CSV | `05088f11c7929995ef6dd28e5135669ebaf37465928c62c826a15f245520299a` |
| BTC source manifest | `76bbda3bef3be823324f237b5f95354ea55b0e75085a5d9d52b64351fe5f0f5f` |
| BTC quality report | `f5887a4d0554dd89a3338af12caa09e35f009a806f3ee097636e6caba22e790a` |
| BTC Development origin plan | `c8737fe917c830e6fb853343e684221d9f0c5543cd84807b92b79e7ee75aaf90` |
| BTC Validation plan (identity hash only) | `2a023e69a46aa952ca0015c2d8a87574616366a016c5cad333204418bda12e84` |
| SPY Development report (read-only comparator) | `972a196f7beae4e31a343a1163beb678f76ee29e63362bb9b10171d3ea1ff149` |

BTC preflight hashes the complete immutable dataset CSV for integrity but parses **only the first 1963 rows** (2017-08-17 through 2022-12-31). It never parses or evaluates the BTC Validation target values or any Final Test data. The pre-published Validation plan is only hash-checked.

**Strict information set:** completed day-`t` close and all daily returns through `t` only. `context = (r_{t-255},...,r_t)` with `r_t=log(C_t/C_{t-1})`; predict `r_{t+1}` then observe the target for calibration. No calendar/business-day resampling is performed on BTC. The existing Moirai adapter's synthetic GluonTS BDay labels are model interface metadata, not BTC market timestamps; the actual BTC timestamps and horizon are enforced outside the adapter.

## Forecast execution

- 1461 Development targets/model (2019-01-01 through 2022-12-31).
- First origin is **2018-12-31** with absolute BTC row index `501`; first seed is `2528 = 2027 + 501`. Last origin is 2022-12-30, index `1961`.
- Models run **sequentially** in `.venv` (Chronos) and `.venv-moirai` (Moirai), `horizon=1`, `context=256`, `num_samples=100`, sample quantiles 0.1/0.5/0.9. Checkpoint commit and weight SHA must match previously accepted audit.
- Gate A: 8 contiguous Development records/model, first 2 per model deterministic replay. Validate every record (seed, date, target, context SHA, quantiles, timing and provenance), plus cross-model alignment. **Only after Gate A PASS** proceed to Gate B full cache.
- Gate B: 1461 contiguous records/model; atomic write-once per-origin JSON, aggregated CSV and canonical manifest with all per-origin SHA-256. No clipping, interpolation, or silent replay on existing records.
- Resume verifies all existing published records as a contiguous prefix, rejects mismatches and out-of-plan JSON; failures preserve records. `.btc-dev.lock` is exclusive. Inspect process/lock owner before handling a stale lock.

## CPU-only calibration and evaluation

Reuse `Parameters()` and `calibrate()` unmodified: decimal log returns, alpha=0.20, W=252, gamma0=0.01, EWMA decay=0.94, shared return-EWMA driver clipped [0.5,2.0], expansion-only correction. BTC initializes an **independent causal calibration state** from 256 returns through its first origin. The first 252 Development forecasts provide sequential calibration feedback; 1209 subsequent origins are evaluated in each of 9 factorial conditions/model. The signed-score rolling CQR ablation is reported separately (Winkler undefined if it produces an empty set).

Raw q10/q90 baseline, Winkler, coverage, width, lower/upper misses and penalties, 50-origin rolling empirical coverage and causal high/other volatility-regime diagnostics are generated. Regimes use current origin return EWMA sigma strictly before the next target, compared to the prior 252 origins' 75th percentile. All CIs are **exploratory**, paired circular moving-block percentile 95%, 1000 draws with block lengths 10/20/40; no multiplicity correction. Per-regime intervals use block 20 and count empty replicates explicitly. These regimes do not provide conditional-coverage guarantees.

Read-only comparison to SPY Development reports **within-dataset paired differences**, never absolute cross-market Winkler as comparable loss magnitudes. No model selection or hyperparameter tuning is performed. Do not run BTC Validation in this task.

## Output paths and scientific risk flags

All new artifacts live beneath:

- `artifacts/forecast_cache/btcusdt/development/gates/<BTC_DEV_PLAN_SHA>.json` — write-once Gate A verification report with first-8 per-model hashes
- `artifacts/forecast_cache/btcusdt/development/runs/<BTC_DEV_PLAN_SHA>/<model>/records/*.json`
- `artifacts/forecast_cache/btcusdt/development/runs/<BTC_DEV_PLAN_SHA>/<model>/predictions.csv`
- `artifacts/forecast_cache/btcusdt/development/runs/<BTC_DEV_PLAN_SHA>/<model>/manifest.json`
- `artifacts/calibration/btcusdt/development/<BTC_DEV_PLAN_SHA>/development_1209_per_cell.csv`
- `artifacts/calibration/btcusdt/development/<BTC_DEV_PLAN_SHA>/btc_development_calibration_report.json`
- `artifacts/calibration/btcusdt/development/<BTC_DEV_PLAN_SHA>/task008b_spy_btc_contrasts.json`

Keep `contamination_free_verified=false`, `btc_historical_point_in_time_verified=false`, and `scientific_clearance=false`. A successful technical audit is NOT a scientific clearance. BTC Validation execution remains blocked until this task's local logs and immutable outputs are independently reviewed.

## Single Windows execution command

Run from project root after downloading ZIP to Downloads:

```powershell
& {
    $ErrorActionPreference = 'Stop'
    Expand-Archive -Path "$HOME\Downloads\TASK-008B-btc-development-patch.zip" -DestinationPath '.' -Force -ErrorAction Stop
    $env:PYTHONPATH = @((Resolve-Path .\src).Path, (Resolve-Path .\scripts).Path) -join [IO.Path]::PathSeparator
    & .\.venv\Scripts\python.exe .\scripts\task008b_btc_development.py run --root . --allow-btc-development
    if ($LASTEXITCODE -ne 0) { throw 'TASK-008B FAIL — preserve BTC records and logs for resume; do not delete artifacts.' }
}
```

The CLI automatically runs only TASK-008B focused tests, then BTC input/checkpoint preflight, small gate, full inference, CPU analysis and report audit. It does not rerun SPY audits. On interruption, repeating the identical command resumes existing records. No claims of successful Windows inference or real BTC metrics are made before receiving the execution log.
