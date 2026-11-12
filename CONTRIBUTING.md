# Contributing Guidelines

We welcome contributions to the **Asynchronous Cross-Horizon Probabilistic Adaptation (CITA 2027 Phase 2 / Addendum)** research repository.

## 1. Scientific Principles & Invariants

To maintain absolute scientific rigor, all contributions must uphold these non-negotiable invariants:

1. **Strict Temporal Causality ($\mathcal{F}_t$ Filtration):**
   - Ground truth observation $y_{t+h}$ for horizon $h$ may **never** be accessed or incorporated prior to its causal arrival timestamp $\tau \ge t+h$.
   - Any look-ahead leakage in queues, metric calculations, or online states will fail automated provenance audits (`verify_scientific_integrity.py`).

2. **Invariance of Locked Findings:**
   - The original Phase 2 baseline benchmark findings (0/8 Holm-corrected rejections on SPY and BTCUSDT log returns, 24/24 primary bootstrap confidence intervals containing zero) are locked historical facts and must **never** be retroactively altered or masked.

3. **Scale Positivity & Lyapunov Boundedness:**
   - Multiplicative dispersion scale updates must satisfy $s_{t,h} > 0$ for all $(t, h)$.
   - Online latent states must remain uniformly ultimately bounded in a compact ball $\|\mathbf{z}_t\| \le R$.

4. **Strictly Proper Scoring Rules:**
   - Distributional forecasting quality must be evaluated using strictly proper scoring rules: Continuous Ranked Probability Score (CRPS), Multivariate Energy Score, and Variogram Score.

## 2. Development Setup

```bash
# Clone the repository
git clone https://github.com/vlqm2008/cita2027-phase2.git
cd cita2027-phase2

# Activate Python 3.11 virtual environment
.venv\Scripts\activate   # On Windows
source .venv/bin/activate # On Linux/macOS

# Install in editable mode with development dependencies
pip install -e ".[test,download]"
```

## 3. Running Tests and Integrity Audits

Before submitting a pull request, verify that all test suites pass without warnings:

```bash
# Run complete unit test suite (270 unit tests)
pytest tests/ -v

# Run scientific integrity & causal provenance audit
python scripts/verify_scientific_integrity.py

# Run master end-to-end reproduction suite
python reproduce_all.py
```

## 4. Code Style & Typing

- Python code must comply with PEP 8 and modern Python 3.11+ type annotations (`from __future__ import annotations`).
- Line length is configured to 100 characters in `pyproject.toml`.
- All newly introduced adaptation algorithms must be thoroughly covered by unit tests in `tests/test_*.py`.

## 5. Adding New Adapters or Baselines

When adding a new adaptation algorithm:
1. Implement the adapter class conforming to the `OnlineState` or `BaseAdapter` interface in `src/cita2027/adaptation/`.
2. Register the algorithm in `src/cita2027/adaptation/competitors.py` or the adapter factory.
3. Add full unit tests covering causal state transitions, edge cases (zero feedback, extreme shocks), and metric calculations.
4. Update benchmark scripts and figures if relevant.
