# TASK-011 — Final Test Results & Discussion addendum (English)

The final-test protocol, target dates, model and calibration settings, comparison
family, Holm adjustment, and missing-data policy were locked before 2025–2026
realized targets were decoded by the execution pipeline. A SHA-256 pre-test
manifest was persisted before inference. Forecasts were generated using the
unchanged frozen Chronos-T5-small and Moirai-1.0-R-small adapters. Online
calibration states were carried forward by continuous chronological replay of
Development, Validation, and Final Test records, without reset or weight updates.

## Primary statistical comparisons

For each market and backbone, the two pre-registered comparisons measure the
paired mean Winkler difference between residual- or return-normalized Rolling
CQR and unscaled Rolling CQR. Negative differences favor normalization.
Unadjusted 95% paired circular moving-block bootstrap percentile intervals use
block lengths 10, 20, and 40. The primary confirmatory *decision procedure* is
a one-sided, centered bootstrap test at block length 20, with Holm adjustment
across all eight comparisons. Intervals are not familywise adjusted; they should
not be interpreted as such. Results are reported regardless of sign.

| Market | Model | Scale | Mean Δ Winkler | CI95 B20 | raw one-sided p | Holm p | Holm reject |
|---|---|---|---:|---|---:|---:|---|
| btcusdt | chronos | residual_ewma | -0.0003658 | [-0.0020713, 0.0013375] | 0.35664 | 1.00000 | False |
| btcusdt | chronos | return_ewma | -0.0003731 | [-0.0020739, 0.0013651] | 0.34865 | 1.00000 | False |
| btcusdt | moirai | residual_ewma | -0.0000176 | [-0.0002978, 0.0002847] | 0.47253 | 1.00000 | False |
| btcusdt | moirai | return_ewma | -0.0000693 | [-0.0003337, 0.0001722] | 0.28771 | 1.00000 | False |
| spy | chronos | residual_ewma | -0.0010235 | [-0.0034494, 0.0010432] | 0.17682 | 1.00000 | False |
| spy | chronos | return_ewma | -0.0007943 | [-0.0032065, 0.0012829] | 0.22877 | 1.00000 | False |
| spy | moirai | residual_ewma | 0.0000262 | [-0.0004252, 0.0004294] | 0.54146 | 1.00000 | False |
| spy | moirai | return_ewma | -0.0000025 | [-0.0004931, 0.0003908] | 0.47652 | 1.00000 | False |

## Interpretation and limitations

Differences in mean interval score must be interpreted jointly with marginal
coverage, width, lower- and upper-tail misses, and volatility-regime diagnostics.
The fixed causal high-volatility regime indicator is based on pre-origin return
EWMA sigma compared with the preceding 252 origin sigmas; observed rolling
coverage is not a proof of conditional coverage. The eight pre-registered
comparisons control familywise error only under the bootstrap assumptions, not
against checkpoint training contamination or historical data revisions.
Chronos/Moirai test checkpoint training-data overlap remains unverified, Yahoo
adjusted close is retrospectively adjusted, Binance monthly archives may have
been corrected after their event times, and Moirai's synthetic business-day
calendar is not proven semantically equivalent to daily UTC BTC dynamics.
Native intervals and all nine factorial configurations remain disclosed.
Tests were not used to tune hyperparameters, select backbones, or revise the
pre-registered comparison family. The experiment is a controlled empirical
benchmark, not a newly proven conformal algorithm or a trading guarantee.

This addendum supplements TASK-010 and does not overwrite its accepted text.
Scientific clearance remains false.
