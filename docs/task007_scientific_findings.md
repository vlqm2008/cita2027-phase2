# TASK-007 — Frozen scientific finding summary (SPY only)

This document reflects the accepted immutable Development 2019–2022 and Validation 2023–2024 JSONs, not a new inference or calibration run. All results are exploratory, multiple comparisons unadjusted. See separate machine-readable `task007_reference_findings.json` for every factorial cell's Development/Validation metrics and Validation block-length CI signs.

## Replication

- **Winkler, normalized Rolling CQR against unscaled Rolling CQR:** Chronos Residual Δ −0.001242 and Return Δ −0.001194 on Validation; Moirai Residual Δ −0.000297 and Return Δ −0.000319. All four 95% paired bootstrap CIs exclude zero below zero for block lengths 10, 20, 40. The same contrast was favorable (CIs below zero across 10/20/40) in Development corrected diagnostics, supporting *within-sample robustness across these two chronological periods* but not a multiplicity-adjusted or zero-contamination proof.
- **Native q10/q90 differs by model:** Chronos Validation 19.92% coverage, Winkler 0.05177; Moirai Validation 80.48% coverage, Winkler 0.03211. Development Chronos native 14.71%, Moirai native 71.77%; native calibration quality cannot be assumed uniform across frozen TSFMs.
- **Nominal 80% marginal coverage:** Most calibrated cells near 80%; unscaled Rolling CQR Chronos has 85.66% and Moirai 82.87%, which is not the same as conditional coverage.
- **High-volatility undercoverage:** Only 71 high-regime Validation origins (431 other); high-regime Chronos unscaled Rolling 70.42% versus scaled Rolling 83.10% coverage, but Chronos **high lower-tail miss remains 11.27%** across these Rolling conditions; improved coverage is mainly upper-tail. High-regime Moirai unscaled Rolling 69.01%; residual/return Rolling approximately 70.42%. Do not claim reliable downside protection. Group-specific CIs are wide.

## Non-replication and hypotheses

- Variable-gamma ACI minus fixed-gamma ACI: all six Validation same-scale contrasts (3 scales × 2 backbones) have paired 95% CIs containing zero at 10/20/40. Thus evidence for a separate adaptive-gamma gain has **not replicated**. Development gamma driver upper-bound saturation was 61.83%, including all 250 evaluated origins in 2022; a fixed gamma=0.02 sensitivity partially explains original differences. Do not relabel it as a confirmed adaptive effect.
- Higher absolute Winkler in Development than Validation in all 18 factorial cells is **not** a paired cross-period test and may largely reflect differing market volatility. High regimes: Development 290/1006 (28.83%), Validation 71/502 (14.14%). Unscaled-versus-scaled subgroup impacts remain descriptive/exploratory.
- On Validation, Moirai native q10/q90 numerical mean Winkler (0.032110) is below all Moirai calibrated-cell means (range 0.032184–0.032965), but no direct native-vs-calibrator paired CIs are present. Do not assert superiority.

## Boundary and restrictions

- The accepted Validation audit reports 1258 Development records replayed continuously before 502 Validation records with no ACI/EWMA/score buffer reset; the aggregate JSON itself is not an independent per-origin state proof.
- Three scientific flags remain **false**: `contamination_free_verified`, `point_in_time_adjusted_prices_verified`, `scientific_clearance`.
- No method selection is authorized from Validation. SPY Final Test remains untouched.
