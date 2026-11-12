# Asynchronous Cross-Horizon Probabilistic Adaptation for Frozen Time-Series Foundation Models

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![PyTorch 2.6](https://img.shields.io/badge/pytorch-2.6-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub sync](https://img.shields.io/badge/github-vlqm2008%2Fcita2027--phase2-brightgreen)](https://github.com/vlqm2008/cita2027-phase2)

> **CITA 2027 Collaborator Project C+ Addendum**  
> Empirical validation and rigorous statistical testing of asynchronous cross-horizon probabilistic adaptation for frozen time-series foundation models (Chronos-T5-small & Moirai-1.0-R-small) under distribution shift.

---

## 1. Overview & Research Problem

Time-series foundation models (TSFMs) deployed in non-stationary real-world environments cannot adapt their frozen parameters to concept drift. Standard online post-processors treat forecasting horizons independently, waiting for ground truth before updating. However, in multi-horizon forecasting, supervision arrives **asynchronously**: short horizons ($h=1$) realize immediately, while long horizons ($h=16$) incur substantial supervision delays.

This project investigates whether newly observed short-horizon forecast feedback can transfer useful structural information to improve delayed long-horizon predictive distributions, while avoiding negative transfer caused by stale feedback or overlapping autocorrelation.

Full formal paper manuscript: [RESEARCH_PAPER_ADDENDUM.md](docs/RESEARCH_PAPER_ADDENDUM.md).

---

## 2. Research Hypotheses & Findings

| Hypothesis | Description | Empirical Outcome | Key Evidence |
|---|---|---|---|
| **$H_1$: Cross-Horizon Transferability** | Forecast mismatch shares a low-dimensional manifold across horizons; a shared state improves long horizons ahead of delayed ground truth. | **Partially Supported (Regime-Dependent)** | Gains occur primarily under correlated multi-horizon structural shifts (Joint Shift: -21.8% CRPS). In stationary noise, naive coupling causes negative transfer. |
| **$H_2$: Stale-Feedback Negative Transfer** | Incorporating delayed residuals with uniform weights causes feedback latency shocks under abrupt reversals; exponential age decay ($w = e^{-\gamma \cdot \text{age}}$) mitigates degradation. | **Confirmed** | Statistically significant reduction in CRPS degradation under abrupt reversal ($\Delta \text{CRPS} = -0.0487$, Holm $p < 0.001$). |
| **$H_3$: Overlap-Aware Stability & Unified Adapter** | Multi-horizon forecast errors from overlapping origins are strongly dependent. Effective-sample-size scaling ($K_{\text{eff}}$) stabilizes adaptation. | **Confirmed** | $H_3$ outperforms independent EWMA ($B_1$) across all long horizons ($h=4, 8, 16$, Holm $p < 0.001$) under joint shift, and significantly outperforms online EWMA on real financial series (SPY). |

---

## 3. Publication Figures

The figures are rendered in 300 DPI publication resolution in `paper_figures/`:
- **Figure 1 (`paper_figures/fig1_joint_shift_crps_by_horizon.png`):** CRPS across horizons $h \in \{1, 4, 8, 16\}$ under severe Joint Shift, demonstrating $H_3$'s growing advantage over frozen ($B_0$) and independent EWMA ($B_1$) as horizon increases.
- **Figure 2 (`paper_figures/fig2_abrupt_reversal_adaptation.png`):** Prequential CRPS trajectory under Abrupt Reversal, showing how $H_2$ and $H_3$ dampen stale-feedback shocks compared to naive uniform updates.
- **Figure 3 (`paper_figures/fig3_cross_regime_benchmark_summary.png`):** Comprehensive radar and bar benchmark across all 8 distribution shift regimes.

---

## 4. Statistical Significance

All paired contrasts were evaluated via **Circular Block Bootstrap** ($B = 1000$, block lengths $b \in \{5, 10, 20\}$) and **Wilcoxon signed-rank tests** with **Holm-Bonferroni step-down correction** at $\alpha = 0.05$.
- Summary tables: `paper_tables/table_adaptation_statistical_tests.csv` and `paper_tables/table_real_data_contrasts.csv`.
- Markdown report: `paper_tables/STATISTICAL_TEST_REPORT.md`.

---

## 5. Quickstart & Reproducibility

### Environment
The project runs with Python 3.11:
```bash
# Activate environment
.venv\Scripts\activate

# Install package in editable mode
pip install -e .
```

### Reproduce Benchmarks
```bash
# 1. Run Hypothesis H1 benchmark (Cross-Horizon Transfer)
python scripts/run_hypothesis_h1.py

# 2. Run Hypothesis H2 benchmark (Age-Decay & Stale Feedback)
python scripts/run_hypothesis_h2.py

# 3. Run Hypothesis H3 unified benchmark (Overlap-Aware Adaptation)
python scripts/run_hypothesis_h3.py

# 4. Run Real-World Market Benchmark (SPY & BTCUSDT, CPU-light)
python scripts/run_real_data_adaptation.py

# 5. Run Paired Statistical Tests (Circular Block Bootstrap & Holm correction)
python scripts/run_statistical_tests.py

# 6. Generate 300 DPI Publication Figures
python scripts/plot_paper_figures.py
```

### Run Unit Tests
```bash
pytest tests/test_adaptation_*.py tests/test_real_data_benchmark.py tests/test_significance.py -v
```
All 15 tests pass deterministically.

---

## 6. Locked CITA Phase 2 Integrity

This project maintains strict isolation from the original locked Phase 2 CITA artifacts:
- Original zero-shot calibration findings (0/8 Holm rejections on Final Test) remain unchanged.
- All Phase 2 manifests, boundary hashes, and caches in `artifacts/calibration/` and `artifacts/reports/` are preserved in read-only lock.
- New adaptation modules and experiments reside independently in `src/cita2027/adaptation/`, `artifacts/adaptation/`, and `paper_figures/`.
