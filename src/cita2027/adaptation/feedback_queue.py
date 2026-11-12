"""Causal asynchronous feedback queue enforcing strict non-leakage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List
import numpy as np


@dataclass(frozen=True)
class ForecastFeedbackItem:
    origin_t: int
    horizon: int
    target_t: int  # origin_t + horizon
    y_true: float
    base_samples: np.ndarray
    adapted_samples: np.ndarray | None = None
    metadata: dict[str, Any] | None = None


class CausalFeedbackQueue:
    """Maintains pending multi-horizon forecasts and releases observations only at target_t."""

    def __init__(self):
        self._pending: list[ForecastFeedbackItem] = []
        self._history: list[ForecastFeedbackItem] = []

    def enqueue(self, item: ForecastFeedbackItem) -> None:
        if item.target_t <= item.origin_t:
            raise ValueError(f"Target timestamp ({item.target_t}) must be strictly > origin ({item.origin_t})")
        self._pending.append(item)

    def step_and_release(self, current_t: int) -> list[ForecastFeedbackItem]:
        """Release all feedback items that have matured up to current_t.

        Strict invariant: An item for origin t and horizon h is released IF AND ONLY IF
        current_t >= t + h.
        """
        ready: list[ForecastFeedbackItem] = []
        remaining: list[ForecastFeedbackItem] = []

        for item in self._pending:
            if item.target_t <= current_t:
                ready.append(item)
                self._history.append(item)
            else:
                remaining.append(item)

        self._pending = remaining
        # Return sorted by target_t, then horizon
        ready.sort(key=lambda x: (x.target_t, x.horizon))
        return ready

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def released_count(self) -> int:
        return len(self._history)
