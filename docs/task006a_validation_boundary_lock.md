# TASK-006A — Validation boundary protocol (locked planning draft)

Status: **PLANNING-ONLY IMPLEMENTATION; LOCAL WINDOWS PLAN HASH & GPU EXECUTION NOT YET AUDITED**.
This is an additive patch to the accepted `cita2027-phase2` repository. Do not modify original TASK-004/005 source or artifacts. Validation and final-test outcome access is prohibited until separately authorized.

## 1. Temporal contract and eligibility

- Dataset: same processed SPY snapshot SHA `4375f3b90c5dbfe1cfd1a8191a36c15c2d878993ef04b130c8de8810927103ba`.
- The accepted Development plan covers origins 3272..4529, targets 3273..4530, and the last forecast has origin 2022-12-29 -> target 2022-12-30.
- Validation targets are exchange-row indices **4531..5032**, all 502 observations of the 2023–2024 Validation split. Forecast origins **4530..5031**; origin 4530 (2022-12-30) forecasts the first validation target 4531 (2023-01-03) and is labeled `boundary_bridge`. All remaining origins belong to Validation.
- A validation session's log return is known only *after* that session closes. At origin `i`, context is exactly `r[i-255:i+1]` (256 historical/present known returns); target is `r[i+1]`, which is prohibited as model input. Realized target enters only post-hoc scoring and next-step updates.
- No origin 5032 forecasts target 5033 (the first Final-Test session). Target 4530 is *not* evaluated in Validation (it is already the last Development target).
- Validate exchange-row adjacency, origin/target dates, previous-date and previous-row references, context SHA-256, and deterministic seed `2027+origin_index`.
- Different backbones use identical origin plan and targets. Chronos runs in `.venv`; Moirai in `.venv-moirai`, sequentially, not concurrently. Models stay frozen; horizon 1; context 256; 100 samples, q10/q50/q90 and exact audited checkpoint revisions.

## 2. Causal state carry-forward

- ACI does **not** reset at the boundary. Signed CQR conformity scores and rolling 252-score buffer carry forward; scores use forecast-origin volatility and are appended **after target observation**.
- Return- and residual-EWMA variance states at the first Validation bridge origin incorporate every completed Development target through 2022-12-30, and no later returns. The return-EWMA shared gamma driver uses this same origin-known state.
- Online calibrator is the unmodified `calibrate()` from TASK-005A. In absence of a separately published immutable state snapshot, `run` reconstructs its state by deterministic **CPU-only sequential replay** of the accepted 1258 Development raw forecasts followed by the 502 Validation forecasts. This does not run or re-fit the original foundation models.
- Every Development replayed `lower`, `upper`, `scale`, `alpha_t`, `gamma_t` is compared at tolerance `1e-12` with the hash-verified TASK-005B per-origin trace; abort on mismatch. It then scores exactly 502 Validation target origins, never resetting the burn-in counter.
- For a prediction at origin `i`, the ACI feedback/error from target `i+1` must not alter its interval; the update may only affect origin `i+1` and later.

## 3. Frozen calibration and analysis

- Nine factorial cells per model: scales `none`, `residual_ewma`, `return_ewma`; controllers `rolling_cqr`, `fixed_gamma_aci`, `variable_gamma_aci`. No Development winner filtering, no hyperparameter tuning on Validation. Do not select a winner at all in this Validation evaluation pass; **0 model-selection steps permitted**.
- Locked constants: target miscoverage `0.20`, rolling score W `252`, gamma0 `0.01`, EWMA decay `0.94`, `min_scale=1e-6`, driver bounds `[0.5,2.0]`, expansion-only correction. Controller alpha boundary `[1/253,252/253]` and signed score quantile `k=ceil((W+1)*(1-alpha_t))`, with finite-sample order-statistic ties retained. Native q10-q90 baseline and standard signed-CQR ablation reported separately; empty signed intervals imply Winkler undefined, never silently removed.
- Measures: Winkler (primary), coverage, width, lower/upper misses, tail penalties and moving rolling empirical coverage (not conditional coverage). Regime classification: rolling 0.75 quantile of the **previous 252 origin-known return EWMA volatility estimates**, equality in `other`; retain the complete Development volatility history across the split; do not reset or retune cutoffs. Regime summaries include n and avoid claiming conditional-coverage guarantees. Paired circular moving-block exploratory bootstrap for comparisons, lengths 10/20/40, 1000 reps, seed 2027, unadjusted 95% percentile CIs and no multiple-comparison correction. Main comparison no-scaling rolling; direct variable-minus-fixed within scaling also computed.
- Final Test and BTC/USDT untouched. Neither Development effect estimates nor Validation results are allowed to select/tune the 18 main methods.

## 4. Read/write contract, execution gate, provenance

- Run `python scripts/task006a_validation.py plan` in `.venv` to do **CPU-only** preflight of frozen sources and accepted Development cache; generate and immutably publish the Validation-origin plan at `artifacts/forecast_cache/validation/plans/<sha>.json`. The command prints the **real local** plan SHA, 502/model (1004 total), bridge date and forecast-only runtime extrapolation based on measured accepted Development runtimes (model loading, extra checks, data access, environment drift and GPU contention EXCLUDED; this is NOT a wall-clock promise).
- Run `python scripts/task006a_validation.py run --allow-validation` **only in a separately authorized later task**. A single command runs focused tests; sequential model-specific GPU worker processes; writes individual immutable checkpoint-bound forecast records and manifests to a new `artifacts/forecast_cache/validation/runs/<validation_sha>/<model>/`; CPU replay/calibration/analysis; immutable outputs to `artifacts/calibration/validation/<validation_sha>`. Resumption loads the exact contiguous prefix and verifies every existing record before completing missing rows. Completed output files are write-once or verify-on-reentry; mismatches halt rather than overwrite. Individual temporary stages may be discarded only under per-model run lock.
- The Validation plan includes complete context hashes. Those hashes are constructed from full processed data for checking alignment; they are NEVER used as input to model inference. The model input is strictly the 256-return context at each origin.
- Persistent risk flags: `contamination_free_verified=false`, `point_in_time_adjusted_prices_verified=false`, `scientific_clearance=false` in each new artifact. Do not interpret technical file audit as scientific clearance.

## 5. Explicit risks / gates

- **BLOCKER for declaring TASK-006A plan-file PASS**: actual processed SPY source, local checkpoint audit and original cache manifests are not mounted in the assistant's runtime, so actual plan SHA and runtime extrapolation must be generated/confirmed by the Windows `plan` command. Do not invent either.
- The accepted TASK-005A calibrator stores no serialized state. Continuous CPU replay plus byte-verified prior Development trace is the state-transfer mechanism; its correctness is conditional on the established calibration recursion and source provenance.
- **RISK**: Original processed adjusted prices are not certified point-in-time; frozen foundation checkpoints are not certified contamination-free. Do not treat validation as an external unbiased benchmark until provenance concerns are separately resolved.
- **RISK**: Time dependency and multiple unadjusted comparisons make validation CIs exploratory. No parameter modification should occur after viewing Validation outcomes.
