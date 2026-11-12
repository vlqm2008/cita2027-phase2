# TASK-011 — Final Test Protocol Lock and Frozen Execution

## 0. Scope and scientific state

**No Final Test was accessed or run while preparing this patch.** All empirical
Final Test results remain **PENDING** until a successful, auditable Windows run.
The accepted SPY/BTC Development and Validation artifacts are read-only.
`contamination_free_verified=false`, `point_in_time_adjusted_prices_verified=false`,
`btc_historical_point_in_time_verified=false` and `scientific_clearance=false` are
immutable risk declarations, not success conditions that may be silently promoted.

A single executable command performs a two-phase operation. With explicit
`--allow-final-test`, successful Phase A automatically authorizes Phase B. Do not
run that command until the intended preregistered holdout evaluation is approved.

## A. Pre-test: strict metadata-only lock

Before inspecting any post-2024 realized return, the runner:

1. verifies accepted TASK-007/TASK-009 evidence anchors, dataset/plan hashes,
   TASK-008D boundary report, accepted source and calibration code hashes;
2. verifies both *actual local checkpoint weight hashes*, each in its accepted
   Python environment, **without GPU inference**;
3. verifies SPY's accepted processed vintage SHA. It reads only its `date`
   column for XNYS session coverage from **2025-01-02** to **2026-10-30**;
4. obtains **2025-01 through 2026-10** BTCUSDT monthly Spot 1d archives,
   verifies Binance `.CHECKSUM` files and does a timestamp-only UTC calendar
   scan for **2025-01-01 through 2026-10-30**, including weekends; October 31
   appears in the archived month but is **excluded** from model evaluation;
5. registers exactly **459 SPY** and **668 BTC** target rows per model, with
   absolute source indices (SPY first target 5033, BTC first target 2694),
   locked origin/target pairs and seeds; no future-dependent context hash is
   generated in Phase A;
6. hashes all TASK-011 runner files as part of the manifest to prevent mutable
   reporting or implementation changes between freeze and execution;
7. stores write-once SHA-named source origin plans and the SHA-named immutable
   pre-test report in `artifacts/final_test/task011/locked/`. It reopens and
   SHA-verifies the published manifest **before Phase B**.

**Privacy / temporal caveat:** Opaque SHA-256 checks necessarily read bytes from
frozen files that may contain realized targets, and the metadata pass reads
calendar fields, but no price, log return, forecast score, or test statistic
is decoded, calculated or inspected until after the pre-test lock. This is a
research-process holdout lock, not cryptographic proof of the researcher's
independence from any earlier access. Preserve the manifest as external evidence.

**Binance version change:** Official Binance public-data documentation states
that Spot archives since 2025-01-01 carry **microsecond** timestamps, not the
millisecond fields used for the accepted 2017–2024 snapshot. Phase A rejects
incorrect timestamp widths. Phase B converts these explicitly and checks UTC
midnight/day ends before computing log returns; old ingestion code and frozen
Moirai adapter remain unchanged. Official reference:
https://github.com/binance/binance-public-data/blob/master/README.md

**Fail closed:** Missing SPY XNYS session, missing Binance month, incorrect
checksums, stale code, unavailable local checkpoint, wrong first/last boundary,
duplicated dates, absent data, or ambiguity over source coverage results in
`NO-GO` prior to target decode; no interpolation, forward fill, replacement
vendor, skipped origin or auto patch is permitted.

## B. Inference, continuous calibration and reporting

- The frozen Chronos/Moirai adapters predict from exactly 256 observed
  decimal daily log returns through forecast origin `i`, never from row `i+1`.
  `seed=2027+i`; horizon=1; 100 samples; empirical q10/q50/q90 (`linear`).
- Each model uses its accepted `.venv` or `.venv-moirai` and pinned commit SHA
  plus revalidated local checkpoint weight bytes. GPUs load sequentially.
- Every forecast record is written atomically and immutable. Resume verifies
  the complete existing contiguous prefix and *never* overwrites original
  records. A failure halts; no skip and no results with incomplete caches.
- The calibrator consumes the previously accepted Development and Validation
  forecasts plus the Final Test forecasts **as one chronological chain**.
  Original Development/Validation per-origin calibration traces are checked
  against the replay output; no reset at either boundary.
- Exactly 3 scaling factors (`none`, residual EWMA, return EWMA) × 3 controllers
  (Rolling CQR, fixed gamma ACI, variable gamma ACI) are evaluated per model
  per market. Constants: W=252, alpha=.20, gamma0=.01, EWMA=.94,
  gamma driver from return EWMA, clip [0.5,2.0], min scale 1e-6, expansion-only.
- Causal high-volatility indicator: return EWMA sigma at origin compared with
  linear q75 of the **previous 252 origin sigmas**. Ties are "other".
- Marginal coverage, width, Winkler and lower/upper tail misses and penalties
  are reported for all 36 factorial configurations and native q10/q90 for each
  market/model; signed-CQR is a separately labeled diagnostic, with empty sets
  scored as undefined for Winkler, not silently removed.
- Results include per-origin traces, paired CIs, Holm primary comparisons,
  secondary variable-vs-fixed comparisons, gamma saturation and high/other
  regimes. The final independent CPU audit recomputes native interval scores,
  regime-weighted arithmetic, all eight paired bootstrap CIs and one-sided
  p-values, and Holm decisions from immutable traces. Published files are
  write-once; accepted manuscripts are unchanged.

## Locked primary multiple-comparison policy

**Eight pre-registered directional hypotheses:** markets {SPY,BTCUSDT} ×
backbones {Chronos,Moirai} × normalization {residual EWMA, return EWMA}:
`mean(Winkler_normalized_rolling - Winkler_none_rolling) < 0`.
The primary model is evaluated on **paired origin-matched differences**.

- **Primary test:** one-sided paired circular moving-block bootstrap with
  **B=20 observations**, **R=1000** and seed **2027**. The one-sided null
  distribution is created by recentering the bootstrap replicate mean
  differences around the observed mean. Finite-replicate correction:
  `(1 + sum(bootstrap_mean-centered <= observed_difference)) / (R+1)`.
- **Multiplicity:** 8 pre-registered p-values with **Holm-Bonferroni FWER
  alpha=.05**, including both markets and both backbones. All eight are reported
  regardless of sign; no method is selected after viewing results.
- **Sensitivity / uncertainty:** 95% paired circular block-bootstrap percentile
  CIs for **B=10/20/40**. These are **unadjusted intervals**, not Holm-adjusted
  confidence bands. No outcome-dependent switching of primary B is allowed.
- **Secondary:** variable gamma vs fixed gamma, native baselines, signed CQR,
  regime differences, downside misses, gamma saturation and bootstrap block
  sensitivity are exploratory. They are reported without confirmatory claims.
- Bootstrap validity under complex nonstationary dependence is assumed for this
  approximate inferential procedure, not a coverage theorem.

## C. Artifact locations (not a new repo)

- `artifacts/final_test/task011/blind_sources/binance_monthly/` — cached
  22 checksummed immutable monthly ZIPs and `.CHECKSUM` files.
- `artifacts/final_test/task011/locked/plans/{spy,btcusdt}_<sha>.json`
- `artifacts/final_test/task011/locked/<pretest_sha>.json`
- `artifacts/final_test/task011/forecast_cache/<pretest_sha>/<market>/<model>/`
  — records, predictions.csv and manifest.json.
- `artifacts/final_test/task011/evaluation/<pretest_sha>/`
  — `final_test_report.json`, `final_per_origin_18cell.csv`,
  `full_factorial_40_rows.csv`, `high_other_regime_80_rows.csv`,
  `primary_8_contrasts_holm.csv`, `TASK-011_Results_Discussion_Addendum.md`,
  `evidence_<sha>.json`.

## D. GO / NO-GO

| Gate | Rule |
|---|---|
| Prelock | GO only when metadata coverage, anchors, checkpoint files, code SHA and exact origin lists all pass |
| Inference | GO only after a durable SHA-verified pretest lock; halt on failure |
| Calibration | GO only with complete origin-aligned forecast manifests for all 4 model-market groups |
| Audit | GO only with valid 36-cell metrics, 8 Holm primary contrasts and trace arithmetic |
| Evidence freeze | GO only with all final cache record SHAs, manifests, reports and exact pretest SHA linked |
| Scientific clearance | **NO-GO**, regardless of technical PASS, while provenance is unresolved |

**Known limitations:** checkpoint training-data contamination, retrospective
SPY adjusted-close vintage, BTC exchange archive vintage, Moirai's synthetic
`BDay` temporal features, possible dependence-bootstrap approximation errors,
multiple testing outside the eight primary contrasts, and imperfect conditional
coverage. The experiment is not a new conformal method proof or a trading test.

## E. Science/research status

A Windows technical PASS validates only the implementable pretest gate, pinned
source coverage, model cache records, trace arithmetic and immutable artifacts.
It does **not** establish an independent third-party audit, out-of-pretraining
model generalization, historical point-in-time reproducibility, conditional
coverage, or general trading usefulness. No parameter tuning is permitted after
any Final Test result is observed, even if results are unfavorable.
