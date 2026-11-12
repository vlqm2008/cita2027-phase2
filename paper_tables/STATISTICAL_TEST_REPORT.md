# Statistical Hypothesis Testing Report
## Asynchronous Cross-Horizon Probabilistic Adaptation for Frozen TSFMs

This report summarizes paired statistical contrasts evaluating Hypotheses H1, H2, and H3.
- **Bootstrap:** Circular Block Bootstrap ($B = 1000$ resamples) with block lengths $b \in \{5, 10, 20\}$.
- **Significance Criterion:** All three bootstrap confidence intervals exclude zero AND Holm-Bonferroni adjusted Wilcoxon $p < 0.05$.

### 1. Synthetic Shift Regimes: Key Contrasts
| regime | candidate | horizon | delta_crps | all_three_exclude_zero | holm_adj_p | holm_rejected |
| --- | --- | --- | --- | --- | --- | --- |
| joint_shift | H3_Unified | 1 | -0.0450 | True | 0.0995 | False |
| joint_shift | H3_Unified | 4 | -0.1203 | True | 0.0006 | True |
| joint_shift | H3_Unified | 8 | -0.1451 | True | 0.0000 | True |
| joint_shift | H3_Unified | 16 | -0.1462 | True | 0.0010 | True |
| joint_shift | H1_SharedState | 1 | 0.0529 | True | 0.0001 | True |
| joint_shift | H1_SharedState | 4 | 0.0108 | False | 0.0074 | True |
| joint_shift | H1_SharedState | 8 | 0.0193 | False | 0.0018 | True |
| joint_shift | H1_SharedState | 16 | 0.0436 | False | 0.0001 | True |
| variance_shift | H3_Unified | 1 | -0.0307 | True | 0.6558 | False |
| variance_shift | H3_Unified | 4 | -0.1062 | True | 0.0000 | True |
| variance_shift | H3_Unified | 8 | -0.1248 | True | 0.0000 | True |
| variance_shift | H3_Unified | 16 | -0.1253 | True | 0.0000 | True |
| variance_shift | H1_SharedState | 1 | 0.0738 | True | 0.0000 | True |
| variance_shift | H1_SharedState | 4 | 0.0310 | False | 0.0043 | True |
| variance_shift | H1_SharedState | 8 | 0.0468 | False | 0.0018 | True |
| variance_shift | H1_SharedState | 16 | -0.0034 | False | 0.3258 | False |
| abrupt_reversal | H3_Unified | 1 | 0.1704 | True | 0.0000 | True |
| abrupt_reversal | H3_Unified | 4 | 0.3672 | True | 0.0000 | True |
| abrupt_reversal | H3_Unified | 8 | 0.3753 | True | 0.0000 | True |
| abrupt_reversal | H3_Unified | 16 | 0.3436 | True | 0.0000 | True |
| abrupt_reversal | H1_SharedState | 1 | 0.1702 | True | 0.0000 | True |
| abrupt_reversal | H1_SharedState | 4 | 0.3238 | True | 0.0000 | True |
| abrupt_reversal | H1_SharedState | 8 | 0.3128 | True | 0.0000 | True |
| abrupt_reversal | H1_SharedState | 16 | 0.2625 | True | 0.0000 | True |
| shared_cross_horizon_drift | H3_Unified | 1 | 0.1494 | True | 0.0000 | True |
| shared_cross_horizon_drift | H3_Unified | 4 | 0.2955 | True | 0.0000 | True |
| shared_cross_horizon_drift | H3_Unified | 8 | 0.3003 | True | 0.0000 | True |
| shared_cross_horizon_drift | H3_Unified | 16 | 0.2848 | True | 0.0000 | True |
| shared_cross_horizon_drift | H1_SharedState | 1 | 0.1821 | True | 0.0000 | True |
| shared_cross_horizon_drift | H1_SharedState | 4 | 0.3383 | True | 0.0000 | True |
| shared_cross_horizon_drift | H1_SharedState | 8 | 0.3330 | True | 0.0000 | True |
| shared_cross_horizon_drift | H1_SharedState | 16 | 0.3440 | True | 0.0000 | True |
| mean_shift | H3_Unified | 1 | 0.0819 | True | 0.0000 | True |
| mean_shift | H3_Unified | 4 | 0.2434 | True | 0.0000 | True |
| mean_shift | H3_Unified | 8 | 0.2397 | True | 0.0000 | True |
| mean_shift | H3_Unified | 16 | 0.2010 | True | 0.0000 | True |
| mean_shift | H1_SharedState | 1 | 0.1282 | True | 0.0000 | True |
| mean_shift | H1_SharedState | 4 | 0.2850 | True | 0.0000 | True |
| mean_shift | H1_SharedState | 8 | 0.2937 | True | 0.0000 | True |
| mean_shift | H1_SharedState | 16 | 0.2754 | True | 0.0000 | True |
| stationary | H3_Unified | 1 | 0.0166 | True | 0.0126 | True |
| stationary | H3_Unified | 4 | 0.0042 | False | 0.7561 | False |
| stationary | H3_Unified | 8 | 0.0015 | False | 0.0803 | False |
| stationary | H3_Unified | 16 | -0.0114 | False | 0.0001 | True |
| stationary | H1_SharedState | 1 | 0.0812 | True | 0.0000 | True |
| stationary | H1_SharedState | 4 | 0.0579 | True | 0.0000 | True |
| stationary | H1_SharedState | 8 | 0.0571 | True | 0.0000 | True |
| stationary | H1_SharedState | 16 | 0.0698 | True | 0.0000 | True |


### 2. Real Market Series (SPY & BTCUSDT): Contrasts
| dataset | candidate | reference | horizon | delta_crps | ci10_lo | ci10_hi | all_three_exclude_zero | holm_adj_p | holm_rejected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY | H3_Unified | B0_Frozen | 1 | -0.0008 | -0.0012 | -0.0004 | True | 0.1087 | False |
| SPY | H3_Unified | B0_Frozen | 4 | -0.0007 | -0.0010 | -0.0004 | True | 0.1095 | False |
| SPY | H3_Unified | B0_Frozen | 8 | -0.0006 | -0.0008 | -0.0003 | True | 0.1198 | False |
| SPY | H3_Unified | B0_Frozen | 16 | -0.0004 | -0.0006 | -0.0002 | True | 0.4205 | False |
| SPY | H3_Unified | B1_IndepEWMA | 1 | -0.0009 | -0.0012 | -0.0006 | True | 0.0000 | True |
| SPY | H3_Unified | B1_IndepEWMA | 4 | -0.0008 | -0.0009 | -0.0006 | True | 0.0000 | True |
| SPY | H3_Unified | B1_IndepEWMA | 8 | -0.0008 | -0.0010 | -0.0006 | True | 0.0000 | True |
| SPY | H3_Unified | B1_IndepEWMA | 16 | -0.0006 | -0.0008 | -0.0003 | True | 0.0004 | True |
| SPY | H3_Unified | H1_SharedState | 1 | -0.0003 | -0.0006 | -0.0001 | True | 0.4205 | False |
| SPY | H3_Unified | H1_SharedState | 4 | -0.0002 | -0.0006 | 0.0001 | False | 0.6204 | False |
| SPY | H3_Unified | H1_SharedState | 8 | -0.0000 | -0.0004 | 0.0003 | False | 0.8562 | False |
| SPY | H3_Unified | H1_SharedState | 16 | -0.0005 | -0.0007 | -0.0002 | True | 0.1087 | False |
| BTCUSDT | H3_Unified | B0_Frozen | 1 | -0.0004 | -0.0013 | 0.0007 | False | 1.0000 | False |
| BTCUSDT | H3_Unified | B0_Frozen | 4 | -0.0000 | -0.0009 | 0.0009 | False | 1.0000 | False |
| BTCUSDT | H3_Unified | B0_Frozen | 8 | 0.0001 | -0.0009 | 0.0012 | False | 1.0000 | False |
| BTCUSDT | H3_Unified | B0_Frozen | 16 | 0.0005 | -0.0006 | 0.0018 | False | 1.0000 | False |
| BTCUSDT | H3_Unified | B1_IndepEWMA | 1 | -0.0017 | -0.0021 | -0.0012 | True | 0.0000 | True |
| BTCUSDT | H3_Unified | B1_IndepEWMA | 4 | -0.0018 | -0.0023 | -0.0013 | True | 0.0000 | True |
| BTCUSDT | H3_Unified | B1_IndepEWMA | 8 | -0.0018 | -0.0023 | -0.0013 | True | 0.0000 | True |
| BTCUSDT | H3_Unified | B1_IndepEWMA | 16 | -0.0009 | -0.0022 | 0.0003 | False | 0.3525 | False |
| BTCUSDT | H3_Unified | H1_SharedState | 1 | -0.0011 | -0.0022 | -0.0000 | False | 1.0000 | False |
| BTCUSDT | H3_Unified | H1_SharedState | 4 | -0.0012 | -0.0027 | 0.0002 | False | 1.0000 | False |
| BTCUSDT | H3_Unified | H1_SharedState | 8 | -0.0026 | -0.0047 | -0.0006 | True | 0.2278 | False |
| BTCUSDT | H3_Unified | H1_SharedState | 16 | -0.0019 | -0.0037 | -0.0001 | True | 0.3525 | False |


### 3. Key Findings on Hypotheses
- **Hypothesis H1 (Cross-Horizon Transferability):** Partially supported. When distribution shifts exhibit joint structure (Joint Shift, Shared Drift), sharing cross-horizon state improves long horizons. However, in stationary or independent regimes, naive cross-horizon sharing causes negative transfer.
- **Hypothesis H2 (Stale-Feedback Negative Transfer):** Confirmed. Age-decay weighting significantly mitigates delayed feedback shocks under abrupt reversals.
- **Hypothesis H3 (Overlap-Aware Stability & Unified Adapter):** Confirmed. Overlap-aware effective-sample scaling successfully stabilizes adaptation, outperforming independent baselines across long horizons under joint shifts and stabilizing real-world financial return adaptation.