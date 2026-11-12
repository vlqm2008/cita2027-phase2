# TASK-005B: full development calibration / statistical analysis

## Scope and locked identities

- **Inputs:** read-only, audited `artifacts/forecast_cache/full_development/runs/39c14b72d7bd82b815757147838e45fe3df56bddf4616367139cfd5661bd089f/{chronos,moirai}`. Each model must have exactly 1,258 validated, uniquely indexed forecasts and matching manifests. `prepare` rebuilds and hashes the immutable origin plan and source-data identities. No TSFM GPU inference is invoked.
- Processed SPY SHA-256: `4375f3b90c5dbfe1cfd1a8191a36c15c2d878993ef04b130c8de8810927103ba`.
- Target: decimal log return of next trading session; context ends at forecast origin. Only 1,258 development origins used. The first 252 origins supply observed-score burn-in; all approaches are evaluated on **the same subsequent 1,006 forecast origins**.
- Frozen TASK-005A constants: alpha = 0.20 (nominal 80%), W = 252, EWMA lambda = 0.94, gamma = 0.01, sigma floor = 1e-6, gamma-driver clamp = [0.5,2], per-origin sampling seed was 2027 + origin index, q10/q50/q90 native quantiles, no hyperparameter retuning. Expansion-only is the **main** 3×3 score correction, not equivalent to ordinary signed CQR.
- Original 18 configurations (9 per backbone) are retained. No cell selection based on TASK-005A pilot results.

## Controls and baselines

- **Raw baseline:** TSFM native q10–q90, evaluated on same 1,006 origins (not calibrating).
- **Rolling CQR with signed scores and signed correction:** \(e_i = \max(L_i-y_i,y_i-U_i)\) with \(Q_i\) = 1-indexed \(\lceil(W+1)(1-\alpha)\rceil\)th smallest **prior** 252 scores; interval \([L_i-Q_i,U_i+Q_i]\). If \(L_i-Q_i>U_i+Q_i\), the conformal set is empty; this implementation reports count and complete-sample empirical coverage with empties uncovered. **Ordinary Winkler is undefined for empty sets; no rows silently removed**. Paired Winkler comparison is unavailable if any empty sets occur.
- **Expansion-only rolling CQR:** `none__rolling_cqr` main factorial cell (signed scores, nonnegative correction `max(0,Q)`); included by explicit alias, not computed as a separate redundant method.
- **Factorial:** score normalization none / causal residual EWMA / causal return EWMA crossed with rolling CQR / fixed gamma ACI / shared-return-driver variable gamma ACI; ACI alpha updates strictly after target observation. Rolling scores use normalized score at its **historical origin scale**. Gamma driver always computed from the **same return EWMA** regardless of scaling cell.
- Ties are retained; target exactly at endpoint is covered. Interpolated quantiles are *not* used for conformal threshold. The bootstrap CI uses `numpy.quantile(..., method='linear')`, which is unrelated to the conformal threshold rank.

## Evaluation

- Primary: mean Winkler interval score, decomposed into mean width + lower penalty + upper penalty. Units: decimal log returns. Lower score is better. Report marginal coverage, mean width, lower/upper miss rates, rolling empirical coverage minimum/maximum (50 sessions), and fixed-threshold return-EWMA high/other regime summaries. **These are not conditional-coverage guarantees**.
- ACI: report frequency of alpha being at bounds at prediction and the number of post-feedback raw alpha proposals outside clipping bounds. Report scale/driver/volatility empirical distributions (min, p05, median, p95, max). These are descriptive statistics.
- Pairing: identical origin-target dates across cells and both backbones. Paired *difference* in per-origin Winkler = candidate minus baseline (negative better), compared to raw and expansion-only `none__rolling_cqr`.
- Pre-specified primary block bootstrap: **circular moving blocks of 20 consecutive trading sessions**, 1,000 replicates, RNG seed 2027; sample the **same block-index matrix across all methods and backbones**, and construct percentile 95% confidence intervals for mean differences. Serial dependence may extend beyond 20 sessions; CIs are exploratory, do not adjust for multiple comparisons. Do not declare statistical significance solely on unadjusted CIs or data-driven ranking. Future block-length sensitivity (e.g. 10/40) must be separately labeled, not overwrite primary output.
- Descriptive factorial contrasts per date: for scale S, average over 3 controllers and subtract the none average; for controller C, average over 3 scales and subtract rolling average; interactions are difference-in-differences `(S,C)-(S,rolling)-(none,C)+(none,rolling)`. These describe effects of *algorithmic choices under the fixed design*, not causal effects of market volatility.
- Native q10/q90 baseline and all factorial models share the same evaluation set; if signed contraction gives empties, report them, do not cherry-pick valid-only Winkler.

## Output contracts

One CLI invocation runs TASK-005A/005B CPU tests, rebuilds/validates the immutable plan and all raw forecast records, creates read-only analyses, and emits:

```
artifacts/calibration/task005b/<plan_hash>/
    full_development_1006_per_cell.csv
    full_development_W252_expansion_only.json
```

Output writes are immutable (existing content must match bytes) and SHA256 referenced. CSV records 18×1,006 evaluated intervals, without affecting raw caches. JSON includes all metrics, per-model paired comparisons, block bootstrap settings and confidence intervals, volatility diagnostics, factor contrasts, input manifest/prediction hashes and safety flags. CLI stdout gives a concise summary and output hash. `contamination_free_verified=false`, `point_in_time_adjusted_prices_verified=false`, `scientific_clearance=false`.

## Windows single command

Extract patch into the *existing* repository (additive files only). From the repository root with `.venv` available:

```powershell
& .\.venv\Scripts\python.exe .\scripts\task005b_development.py
```

No CUDA, no checkpoint reload, no new Python packages, no validation/final targets. If tests, cache audit, or artifact integrity fail, the CLI returns nonzero and should be considered **NO-GO**. Full-development statistical results and GO/NO-GO for the next phase require Windows execution logs; the container cannot see local SPY caches.

## Research limits

The 2019 CQR conformal theorem relies on exchangeability assumptions that are not established for SPY. Clipped ACI and volatility normalizations are modified mechanisms; do not inherit original ACI theorems without new analysis. The retrospective adjusted-price series lacks verified point-in-time integrity, and training-data contamination of TSFMs remains unknown. Development findings are exploratory and must not tune against final test. 

Primary literature: Romano, Patterson, Candès (NeurIPS 2019, *Conformalized Quantile Regression*); Gibbs and Candès (NeurIPS 2021, *Adaptive Conformal Inference Under Distribution Shift*). Block-bootstrap method is a fixed-length circular moving-block implementation and **not** the random-length stationary bootstrap of Politis and Romano (1994).
