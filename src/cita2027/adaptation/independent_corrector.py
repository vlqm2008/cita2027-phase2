"""
Independent Horizon-Specific Online Correctors (Baselines B1 & B2).

Implements:
- Baseline B1: Independent Horizon-wise EWMA Mean Correction
- Baseline B2: Independent Horizon-wise Location & Scale Dispersion Tracking

As mandated by Research Handoff:
- Each horizon h adapts purely based on its own matured feedback
- No information sharing between horizons
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from cita2027.adaptation.correctors import (
    BaselineB0_Frozen,
    BaselineB1_IndependentEWMA,
    BaselineB2_IndependentLocScale,
)


class IndependentEWMACorrector(BaselineB1_IndependentEWMA):
    """Alias for Baseline B1 (Independent EWMA bias tracker)."""
    pass


class IndependentLocScaleCorrector(BaselineB2_IndependentLocScale):
    """Alias for Baseline B2 (Independent Location and Scale multiplier tracker)."""
    pass
