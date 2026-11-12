# TASK-006B — Frozen Validation execution and audit

Status: **implementation; not executed on the user's Windows GPU**.

## Exact immutable identities

- Validation plan: `f574fd5be7e6537c263aa8c9a646dbb8676d87a25b38a70c83131cb0e894fe54`.
- Development plan: `39c14b72d7bd82b815757147838e45fe3df56bddf4616367139cfd5661bd089f`.
- Development report SHA: `972a196f7beae4e31a343a1163beb678f76ee29e63362bb9b10171d3ea1ff149`.
- Development trace SHA: `2f6f88345e5808404ddd609b535f64e3c6a64d0ab11c213305e0f9c73cf39e45`.
- Dataset / checkpoint provenance and sample-forecast adapter are reused verbatim from TASK-006A.

## Operational procedure

Invoke `scripts/task006b_validation.py run --allow-validation` in the existing repository; Python interpreter is `.venv` (Chronos). The wrapper verifies the **published exact SHA** and fixed protocol **before any GPU work**, runs only `tests/test_task006b_validation.py`, launches the existing TASK-006A `_worker` sequentially from `.venv` and `.venv-moirai`, then invokes the accepted CPU `evaluate` for state continuity, signed-CQR, 9 factorial cells/model and paired block bootstrap. `audit` is CPU-only and can be re-run without forecasting. `preflight` is CPU-only.

A crash retains write-once JSON records and a subsequent `run` resumes a complete contiguous prefix. If `.validation.lock` remains after a hard process crash, inspect any still-running process first. Do **not** overwrite or delete records or the lock while a process may be active. No changes are made to the TASK-006A script or any accepted Development artifacts.

## Causal information set

For forecast origin index `i`, inference context is return rows `i-255` through `i` inclusive (256), target is `i+1` and is not given to the adapter. The first forecast uses Development last origin `i=4530` and scores Validation target `i+1=4531`. Last target is index `5032`; the Final Test begins at `5033`. Targets already realized after each origin are used for state updates before the *next* origin. Continuous calibration replay over 1,258 Development and 502 Validation records is matched to the SHA-locked Development 1,006-row/cell trace prefix, then scored on exactly 502 Validation origins/cell. No reset of ACI alpha, signed scores, return-EWMA or residual-EWMA states.

## Locked methods/statistics

- Frozen Chronos and Moirai: context 256, horizon 1, samples 100, empirical quantiles 0.1/0.5/0.9, fixed checkpoint revisions, origin-dependent seed `2027 + i`.
- All 9 scaling × controller cells for each model, signed scores, expansion-only main intervals, W252, target miscoverage 0.2, EWMA decay 0.94, ACI gamma base 0.01, shared return-EWMA gamma driver clip [0.5,2]. No winner selection.
- Baselines: raw q10–q90 and rolling signed CQR (empty sets explicit). Interval score, empirical coverage, width, lower and upper misses, empirical rolling coverage, high/other volatility regimes and within-regime score uncertainties.
- Regime labels: origin-time return-EWMA volatility > rolling 75th percentile of the **prior 252 origin volatilities**. Strict >, equality = other. No outcomes after forecast origin used.
- Paired circular moving-block bootstrap: 1,000 replicates, seed 2027 and lengths 10,20,40. CIs are **exploratory and not multiplicity corrected**. Cross-period Development-vs-Validation differences are **descriptive**, not a paired dependence-aware two-sample significance test.
- Note: `same_scale_fixed` comparator is equality (zero difference) for the fixed-gamma row by construction; it is not an independent method.

## Reports and flags

Results published under `artifacts/forecast_cache/validation/runs/<VAL_SHA>/{chronos,moirai}/` and `artifacts/calibration/validation/<VAL_SHA>/`. All paths write once: any changed re-run bytes are rejected. TASK-006B adds `task006b_scientific_comparison.json`, a compact descriptive Development-versus-Validation summary, without altering accepted TASK-006A report schema.

Final Test and BTC/USDT are not accessed. Contamination, point-in-time adjusted-price and scientific-clearance flags remain `false`. The presence of valid technical reports does not constitute scientific clearance or evidence of statistically significant superiority.
