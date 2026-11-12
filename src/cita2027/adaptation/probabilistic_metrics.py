"""
Probabilistic Forecast Verification & Proper Scoring Rules.

Re-exports and unifies proper scoring rules for multi-horizon forecast evaluation:
- Continuous Ranked Probability Score (CRPS)
- Continuous Winkler Interval Score
- Multivariate Energy Score (joint trajectory verification)
- Variogram Score of orders p=0.5 and p=1.0 (correlation structure verification)
- Probability Integral Transform (PIT) Uniformity & Anderson-Darling tests
- Expected Calibration Error (ECE) and Normalized Interval Sharpness
"""

from __future__ import annotations

from typing import Sequence

from cita2027.adaptation.distribution_metrics import (
    CalibrationReport,
    PITDiagnosticResult,
    TrajectoryEvaluator,
    compute_pit_values,
    continuous_winkler_score,
    energy_score,
    evaluate_calibration_and_sharpness,
    pit_anderson_darling_uniformity,
    variogram_score,
    winkler_score_level,
)


def compute_univariate_crps(y_true: float, samples: Sequence[float]) -> float:
    """Standard empirical CRPS for predictive sample ensemble."""
    import numpy as np
    s = np.asarray(samples, dtype=np.float64)
    term1 = float(np.mean(np.abs(s - y_true)))
    term2 = 0.5 * float(np.mean(np.abs(s[:, None] - s[None, :])))
    return term1 - term2


def compute_crps_empirical(samples: Sequence[float], y_true: float) -> float:
    """Alias for empirical CRPS accepting (samples, y_true)."""
    return compute_univariate_crps(y_true=y_true, samples=samples)

