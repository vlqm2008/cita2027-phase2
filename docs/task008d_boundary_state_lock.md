# TASK-008D — BTC Development→Validation Boundary and State Transfer Lock

**Scope:** CPU-only verification, read-only accepted input/forecast artifacts. No new forecast, no recalibration publication, no BTC Validation inference, no final-test files, no SPY writes.

## Immutable anchors

- BTC processed snapshot SHA-256: `05088f11c7929995ef6dd28e5135669ebaf37465928c62c826a15f245520299a`
- BTC Development origin plan SHA-256: `c8737fe917c830e6fb853343e684221d9f0c5543cd84807b92b79e7ee75aaf90`
- BTC Validation origin plan SHA-256: `2a023e69a46aa952ca0015c2d8a87574616366a016c5cad333204418bda12e84`
- Development report SHA-256: `7782834dfa3aaa0e863a82a02c1e49604a493b2be2223467320fb782e0933210`
- BTC–SPY contrasts SHA-256: `47f697cd1ea193d2fc0c78ba53e69158183cb15d612e44f5c89412bcc113060f`

## Boundary specification

Define BTC row index i starting at 2017-08-17, one UTC daily bar per row. Dev ends on 2022-12-31 (index 1962); Validation first target 2023-01-01 (index 1963). Forecast origin index 1962, context indices `[1707,1962]`, horizon 1, seed `2027 + 1962 = 3989`. The entire 731-day Validation plan is checked for UTC date continuity including weekends and holidays. The BTC dataset is NOT indexed by SPY trading days.

## State information set

At the first Validation origin, all 1,461 Development targets through 2022-12-31 have already been observed. For each model × scale × controller, replay the complete Development sequence with the frozen `Parameters()` and verify the last Development interval against the accepted `calibrate()` implementation. The saved boundary state contains the last 252 *normalized signed conformity scores*, two EWMA variances, their initialization reference, and the dynamic ACI alpha for the next origin. Score and variance updates follow prediction→observation→update; alpha is updated only on evaluated rows after the 252-target burn-in. Regime history contains the prior 252 return-EWMA **origin-time** volatility values and compares the first Validation-origin volatility against its historical q75 threshold.

A separate **protocol-only synthetic transfer probe** takes the last three Development forecast quantile triplets and targets and relabels them at the first three Validation origin indices. It verifies score buffers and ACI alpha carry forward without reset, *but is not a BTC Validation TSFM forecast, a valid Validation interval, or a BTC Validation metric*. Neither Validation targets nor Validation forecast records are read in TASK-008D. This limitation is explicit in the report.

## Moirai BDay behavior

The accepted adapter (`Moirai10Adapter`) constructs a **synthetic** `ListDataset` beginning `2000-01-03` using `freq='B'`, with 256 scalar returns and `feat_dynamic_real_dim=0`, `past_feat_dynamic_real_dim=0`. Neither this synthetic start nor generated synthetic future timestamps are mapped onto actual BTC target dates; timestamps and origin alignment are controlled **externally** by the immutable UTC origin plan. The CPU-only Moirai venv probe checks real GluonTS `ListDataset` outputs and absence of supplied covariate fields. It does not prove that all internal versions of Uni2TS/GluonTS cannot generate frequency-derived features after transformation. No adapter changes in TASK-008D; replacing BDay with daily frequency after seeing BTC Development results would change model inputs and invalidate the frozen forecasts. A full source/provenance audit would require inspecting the pinned Uni2TS transform chain or tracing a predictor; this is a remaining risk, not an invented PASS.

## Next step plan: TASK-008E (not executed here)

- Before GPU: verify TASK-008D boundary manifest SHA, original BTC Validation origin plan SHA, BTC Development forecast model manifests and the 9 state hashes/model.
- Reuse frozen `ChronosT5Adapter`, `Moirai10Adapter` without changing BDay behavior; check two-model first-origin output and timestamp mapping in a small gate.
- Produce 731 forecasts/model as an immutable, resumable cache, subject to GPU small-batch PASS.
- Replay **all Development records followed by all Validation records** in the accepted CPU calibrator in one chronological sequence. Do not initialize/reset at Validation. Score Validation 731 origins/cell; compare to reference state at bridge before publication.
- Never use SPY Final Test, BTC 2025+ Final Test, future targets at forecast time, nor use BTC Validation for tuning/winner selection.

The future TASK-008E implementation shall provide a single PowerShell invocation; TASK-008D does **not** ship an untested full-Validation runner and therefore does not advertise a runnable TASK-008E command prematurely.

## PASS semantics

A successful Windows run can assert locked boundary, 731-day plan continuity, BTC cache completeness, causal state replay, and CPU GluonTS input shape. Independent validation of the complete internal Moirai transform chain and checkpoint training-data disjointness remain unresolved. Keep `contamination_free_verified=false`, `btc_historical_point_in_time_verified=false`, `scientific_clearance=false`.
