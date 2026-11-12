# TASK-009 — Cross-Market Results and Discussion (manuscript draft)

**Evidence level:** Source-derived aggregate synthesis from frozen SPY and BTCUSDT Development/Validation reports. BTC record-level integrity verification is **pending Windows execution of the TASK-009 evidence-index CLI**. No Final Test was accessed. Reported effects are exploratory and were not adjusted for multiplicity.

**Setup.** The study evaluates one-step-ahead intervals for daily decimal log returns, comparing frozen Chronos and Moirai foundation-model forecasts. The factorial design crosses scale normalization (none, residual EWMA, return EWMA) with Rolling CQR, fixed-step ACI, and volatility-dependent-step ACI. Nominal miscoverage is 0.20; score window is 252; fixed ACI step is 0.01; EWMA decay is 0.94; variable-step driver is clipped to [0.5, 2.0] times the base step. All reported CQR corrections are expansion-only. The calibration state is carried chronologically from Development into Validation. No model selection or parameter optimization is applied based on BTC or SPY Validation results.

## Main validation effects

Paired contrasts are within-backbone, within-market, within-period mean differences in the Winkler interval score (lower is better). The intervals are **reported** 95% paired circular block-bootstrap percentile intervals using block lengths 10, 20, and 40 and 1,000 replicates; they are not newly estimated here.

| Market | Backbone | Normalization (Rolling vs unscaled Rolling) | Mean Δ Winkler | 95% CI, B=20 | CI excludes zero at B=10/20/40 |
|---|---|---|---:|---|---|
| SPY | Chronos | Residual EWMA | −0.001242 | [−0.002271, −0.000226] | Yes / Yes / Yes |
| SPY | Chronos | Return EWMA | −0.001194 | [−0.002245, −0.000172] | Yes / Yes / Yes |
| SPY | Moirai | Residual EWMA | −0.000297 | [−0.000545, −0.000087] | Yes / Yes / Yes |
| SPY | Moirai | Return EWMA | −0.000319 | [−0.000582, −0.000089] | Yes / Yes / Yes |
| BTCUSDT | Chronos | Residual EWMA | −0.001144 | [−0.003282, +0.000797] | No / No / No |
| BTCUSDT | Chronos | Return EWMA | −0.001179 | [−0.003329, +0.000712] | No / No / No |
| BTCUSDT | Moirai | Residual EWMA | −0.000062 | [−0.000581, +0.000367] | No / No / No |
| BTCUSDT | Moirai | Return EWMA | −0.000110 | [−0.000618, +0.000314] | No / No / No |

**Results.** Across SPY Validation (502 targets), both volatility-normalization mechanisms reduced mean Winkler score relative to unscaled Rolling CQR for both frozen models. All four exploratory paired 95% intervals excluded zero across the three block lengths. In contrast, across BTCUSDT Validation (731 daily targets, including weekends), the four mean contrasts retained the negative sign, but all 12 corresponding intervals included zero. Thus the **direction** of the normalization effect recurred across markets, whereas its apparent statistical stability did not. BTC Development (1,209 scored origins/model) showed negative normalization contrasts and intervals excluding zero for the same four comparisons; these effects weakened substantially in BTC Validation. This across-period attenuation is descriptive only; Development and Validation are not paired samples, and no difference-in-effects hypothesis test was performed. Absolute Winkler scores across SPY and BTCUSDT cannot be compared as if the return distributions or scales were matched.

Variable-step ACI did not exhibit a stable advantage over fixed-step ACI across markets, scales, or backbones. On BTC Validation, Moirai without score normalization yielded a small **positive** variable-minus-fixed mean difference (+0.000252, worse), with CIs above zero at B=10 and B=40 but not at B=20. The disagreement in CI exclusion across block lengths and lack of multiplicity adjustment preclude robust superiority claims in either direction. The complete six variable-minus-fixed comparisons per market and period are provided without ranking in `table_paired_contrasts_40_rows.csv`.

## Native intervals and calibration

The raw 10–90% Chronos intervals severely undercover in both markets: SPY Validation coverage 19.92%, BTC Validation coverage 15.87%. The unscaled Rolling CQR coverage values were 85.66% and 79.62%, respectively. In SPY Validation, raw Moirai coverage was 80.48% and raw Winkler was 0.03211, while unscaled Rolling CQR achieved 82.87% coverage but **higher** mean Winkler (0.03296). BTC Validation raw Moirai coverage was 72.91%, compared with 78.93% after unscaled Rolling CQR; a smaller mean Winkler was observed, but superiority over raw Moirai must be assessed using paired intervals, not only point estimates. A conclusion that all calibrated methods outperform native TSFM intervals is therefore not supported.

## Volatility-regime heterogeneity and downside risk

Volatility regimes use a pre-origin return EWMA volatility estimate against a causal prior 252-observation 75th-percentile threshold. The Validation high/other groups contain 71/431 SPY origins and 170/561 BTC origins. These are empirical stratifications, **not conditional coverage guarantees**. Regime counts need not be 25% exactly because a rolling causal threshold, dependence, and finite-sample effects change the realized proportion.

In BTC Chronos high-volatility origins, unscaled Rolling CQR coverage was 72.94%, width was 0.05465, and lower-tail misses were 11.76%. Residual EWMA scaling raised coverage to 83.53%, widened intervals to 0.07186, and reduced lower-tail misses to 5.29%. Return EWMA gave 82.94% coverage and 5.88% lower-tail misses. In other-volatility BTC origins, however, normalized intervals were narrower and empirical coverage fell from 81.64% to 78.07% (residual) or 77.72% (return), with increased lower-tail miss rates. The gain therefore reflects a width and miss-penalty redistribution rather than uniform coverage improvement.

BTC Moirai showed smaller high-regime changes: unscaled Rolling CQR achieved 74.71% coverage and 15.29% lower misses, while return normalization achieved 77.65% coverage and 12.94% lower misses, accompanied by modestly wider intervals. In SPY Validation, high-regime Chronos coverage improved from 70.42% (unscaled) to 83.10% (either scaling), while Moirai improved from 69.01% to 70.42%. High-regime empirical score differences should not be described as statistically significant method-by-regime interactions without *paired interaction* confidence intervals; the reported CIs for individual regime means are insufficient for this inference.

## Interpretation and limitations

**Supported within the frozen aggregate reports.** (i) The factorial evaluations contain 9 calibration cells per backbone for both markets and both chronological periods; (ii) the four SPY Validation normalization contrasts have exploratory CIs below zero at all three tested block lengths; (iii) all four BTC Validation mean normalization contrasts are negative, but every corresponding CI contains zero; (iv) the high-volatility width–coverage tradeoff differs by backbone and regime; and (v) variable ACI learning rates do not reliably dominate fixed rates in these analyses.

**Exploratory, not a new theorem.** The observations are consistent with normalization redistributing interval width in response to volatility clustering. They do not establish universally valid conditional coverage, distribution-free guarantees for the modified procedures, or an improvement under all shifts. The bootstrap intervals are exploratory, unadjusted for the large family of comparisons and conditional on the analyzed sample and bootstrap design.

**Provenance and computational limits.** Checkpoint release dates alone do not establish contamination-free TSFM evaluation; model training-data provenance remains unresolved. Binance archive checksums do not establish historical point-in-time availability of every candle, and SPY adjusted-price vintage verification is incomplete. Moirai uses a synthetic business-day (`BDay`) index; external UTC daily target mapping was audited, but equivalence of all internal temporal features to true seven-day crypto time has not been established. Aggregate reports cannot independently prove per-origin causality, preservation of all score buffers/alpha states, or source record-level integrity; those require accepted record manifests and trace audit. Results must not be used to infer trading profitability or guaranteed tail-risk reduction.

## Scientific-claims register

| Claim | Label | Permitted manuscript wording |
|---|---|---|
| All four SPY Validation normalization contrasts improve mean Winkler with exploratory CIs below zero (B10/20/40). | SUPPORTED (source-report level) | “We observed negative paired differences across all bootstrap block lengths, without multiplicity adjustment.” |
| BTC Validation normalization has the same sign as BTC Development for the four Rolling contrasts. | SUPPORTED descriptively | “The direction of the mean contrasts was retained.” |
| BTC Validation normalization is statistically superior to unscaled Rolling CQR. | UNSUPPORTED | “All four 95% confidence intervals included zero at every tested block length.” |
| Variable gamma outperforms fixed gamma across methods/backbones/markets. | UNSUPPORTED | “No consistent advantage was observed.” |
| Volatility normalization reduces downside exceedances uniformly. | UNSUPPORTED | “Regime- and backbone-specific lower-tail miss rates showed heterogeneous changes.” |
| Score normalization and variable ACI steps are complementary. | EXPLORATORY | “Factorial interactions require direct pre-specified contrasts and adjusted uncertainty analysis.” |
| Improved empirical high-regime coverage implies conditional coverage. | UNSUPPORTED | “High-regime empirical coverage does not establish conditional coverage.” |
| Frozen checkpoints are contamination-free and all historical prices are point-in-time verified. | UNKNOWN | “Training-data overlap and historical data-vintage risks remain unresolved.” |
| Moirai synthetic BDay features are equivalent to true BTC daily features. | UNKNOWN | “Temporal-feature equivalence has not been established.” |
| BTC record-level and state provenance are fully independently verified. | PENDING WINDOWS EVIDENCE INDEX | “Will be asserted only after TASK-009 index execution log and manifest review.” |

## Final Test readiness checklist — NO execution authorized

- [ ] Confirm immutable SPY TASK-007 and BTC TASK-009 evidence manifests are retained with exact SHA-256.
- [ ] Freeze tested code revision, checkpoint identities, data provenance and all 18 configurations; forbid post-Validation selection or tuning.
- [ ] Confirm strict chronological embargo of both Final Test periods and no test data consumption by prior analysis or model selection.
- [ ] Preserve independent evidence of forecast origin alignment, next-day target causality, UTC/weekend calendars and continuous calibration state.
- [ ] Retain a complete comparison family and statistical-analysis plan, including multiple-comparison reporting and block-bootstrap sensitivity.
- [ ] Escalate unresolved checkpoint training-overlap, SPY data vintage, BTC historical point-in-time, and Moirai synthetic-calendar risks as explicit scientific limitations.
- [ ] Obtain separate explicit Final Test execution authorization only after a scientific readiness review; this checklist is **not** authorization.

**Final scientific clearance:** `false`. `contamination_free_verified=false`; `point_in_time_adjusted_prices_verified=false`; `btc_historical_point_in_time_verified=false`.
