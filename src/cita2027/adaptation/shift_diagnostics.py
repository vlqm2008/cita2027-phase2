"""
Distribution Shift Diagnostics & Negative Transfer Boundary Analysis.

Provides diagnostic utilities to detect:
1. Structural breaks and volatility regime switches
2. Adaptation state stability and gradient norm spikes
3. Negative transfer severity (Ratio of adapted CRPS to frozen B0 CRPS)
4. Feedback age distribution and staleness inflation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class NegativeTransferDiagnostic:
    """Diagnostic metrics evaluating presence and severity of negative transfer."""
    regime_name: str
    horizon: int
    b0_crps: float
    adapted_crps: float
    has_negative_transfer: bool
    relative_degradation_pct: float
    diagnostic_verdict: str


def evaluate_negative_transfer_severity(
    b0_crps_list: List[float],
    adapted_crps_list: List[float],
    horizons: List[int],
    regime_name: str = "Regime",
) -> List[NegativeTransferDiagnostic]:
    """
    Check for negative transfer: adapted error > baseline frozen error.
    
    Returns structured diagnostic list per horizon.
    """
    diagnostics = []
    for h, b0, ad in zip(horizons, b0_crps_list, adapted_crps_list):
        is_neg = ad > b0
        deg_pct = ((ad - b0) / b0) * 100.0 if b0 > 0 else 0.0

        if is_neg and deg_pct > 5.0:
            verdict = f"SEVERE NEGATIVE TRANSFER (+{deg_pct:.1f}% error inflation)"
        elif is_neg:
            verdict = f"MILD NEGATIVE TRANSFER (+{deg_pct:.1f}% error inflation)"
        else:
            verdict = f"POSITIVE TRANSFER ({-deg_pct:.1f}% error reduction)"

        diagnostics.append(
            NegativeTransferDiagnostic(
                regime_name=regime_name,
                horizon=h,
                b0_crps=b0,
                adapted_crps=ad,
                has_negative_transfer=is_neg,
                relative_degradation_pct=deg_pct,
                diagnostic_verdict=verdict,
            )
        )
    return diagnostics


def detect_cusum_structural_break(
    residuals: np.ndarray,
    threshold: float = 3.0,
    drift_allowance: float = 0.5,
) -> List[int]:
    """
    CUSUM algorithm to detect timesteps where structural breaks or concept drift occurred.
    """
    s_pos = 0.0
    s_neg = 0.0
    breaks = []
    res_std = float(np.std(residuals)) + 1e-6
    std_residuals = residuals / res_std

    for t, e in enumerate(std_residuals):
        s_pos = max(0.0, s_pos + e - drift_allowance)
        s_neg = max(0.0, s_neg - e - drift_allowance)
        if s_pos > threshold or s_neg > threshold:
            breaks.append(t)
            s_pos = 0.0
            s_neg = 0.0

    return breaks
