"""
Active Information-Theoretic Feedback Selection for Asynchronous Multi-Horizon Adaptation.

In resource-constrained streaming systems (e.g. edge microcontrollers, IoT sensor grids,
high-frequency financial telemetry), acquiring, storing, or verifying ground truth
observations incurs an explicit communication, compute, or query cost c(h).

Active-ACHA formulates feedback ingestion as an online budget-constrained knapsack problem:
    Maximize   Sum_{e in S_t} Value(e)
    Subject to Sum_{e in S_t} Cost(e) <= Budget_t
where Value(e) measures the expected Fisher Information Gain and gradient leverage:
    Value(e) = exp(-gamma * age) * ||w_h|| * |residual(e)| / Cost(e).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from cita2027.adaptation.feedback_queue import ForecastFeedbackItem


class ActiveFeedbackSelector:
    """
    Active Information-Theoretic Feedback Selector (Active-ACHA).
    
    Dynamically prioritizes and sub-samples matured forecast feedback events
    to maximize adaptation velocity under strict bandwidth or compute budgets.
    """

    def __init__(
        self,
        cost_policy: str = "linear_horizon",
        age_decay_gamma: float = 0.08,
    ):
        """
        Parameters
        ----------
        cost_policy : str
            Cost model for horizon h:
            - 'uniform': c(h) = 1.0 for all horizons.
            - 'linear_horizon': c(h) = 1.0 + 0.1 * h (delayed long horizons cost more).
            - 'sublinear': c(h) = 1.0 + 0.5 * log2(h).
        age_decay_gamma : float
            Decay parameter penalizing stale feedback value.
        """
        self.cost_policy = cost_policy
        self.age_decay_gamma = age_decay_gamma

    def compute_cost(self, horizon: int) -> float:
        """Calculate verification cost for observation of lead time horizon."""
        if self.cost_policy == "uniform":
            return 1.0
        elif self.cost_policy == "sublinear":
            return 1.0 + 0.5 * math.log2(max(1, horizon))
        else:  # linear_horizon
            return 1.0 + 0.1 * float(horizon)

    def select_feedback(
        self,
        feedback_items: List[ForecastFeedbackItem],
        current_t: int,
        budget: float = 2.0,
        strategy: str = "active_information_gain",
    ) -> Tuple[List[ForecastFeedbackItem], Dict[str, float]]:
        """
        Select subset S_t of feedback items respecting budget constraint.
        
        Parameters
        ----------
        feedback_items : List[ForecastFeedbackItem]
            Batch of matured feedback events available at time current_t.
        current_t : int
            Current prequential timestamp.
        budget : float
            Maximum allowable cost sum at this timestep.
        strategy : str
            One of ['active_information_gain', 'greedy_short_horizon', 'random_uniform', 'full_unconstrained'].
            
        Returns
        -------
        Tuple[List[ForecastFeedbackItem], Dict[str, float]]
            Selected subset and summary metrics (total_cost, retained_ratio, info_gain).
        """
        if not feedback_items:
            return [], {"total_cost": 0.0, "retained_ratio": 1.0, "total_value": 0.0}

        if strategy == "full_unconstrained":
            cost = sum(self.compute_cost(item.horizon) for item in feedback_items)
            return feedback_items, {
                "total_cost": float(cost),
                "retained_ratio": 1.0,
                "total_value": float(len(feedback_items)),
            }

        scored_items: List[Tuple[float, float, ForecastFeedbackItem]] = []
        for item in feedback_items:
            cost = self.compute_cost(item.horizon)
            age = max(0, current_t - item.origin_t)
            age_weight = math.exp(-self.age_decay_gamma * age)

            base_loc = getattr(item, "base_location", None)
            if base_loc is None:
                base_loc = float(np.median(item.base_samples)) if len(item.base_samples) > 0 else 0.0

            base_disp = getattr(item, "base_dispersion", None)
            if base_disp is None:
                base_disp = float(np.std(item.base_samples)) if len(item.base_samples) > 0 else 1.0

            residual_magnitude = abs(item.y_true - base_loc)
            disp_factor = 1.0 / max(1e-3, base_disp)

            if strategy == "active_information_gain":
                # Value per cost = Fisher information / gradient leverage
                score = (age_weight * residual_magnitude * disp_factor) / cost
            elif strategy == "greedy_short_horizon":
                # Prefer shortest horizon (lowest delay)
                score = 1.0 / float(item.horizon)
            elif strategy == "random_uniform":
                score = float(np.random.rand())
            else:
                score = 1.0

            scored_items.append((score, cost, item))

        # Sort descending by score
        scored_items.sort(key=lambda x: x[0], reverse=True)

        selected: List[ForecastFeedbackItem] = []
        spent_budget = 0.0
        total_value = 0.0

        for score, cost, item in scored_items:
            if spent_budget + cost <= budget + 1e-6:
                selected.append(item)
                spent_budget += cost
                total_value += score

        # If budget is very small and nothing fits, pick at least one minimum-cost item
        if not selected and scored_items:
            min_cost_item = min(scored_items, key=lambda x: x[1])
            selected.append(min_cost_item[2])
            spent_budget += min_cost_item[1]
            total_value += min_cost_item[0]

        retained_ratio = len(selected) / float(len(feedback_items))
        return selected, {
            "total_cost": float(spent_budget),
            "retained_ratio": float(retained_ratio),
            "total_value": float(total_value),
        }
