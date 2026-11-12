"""Prequential simulation engine for Phase 1 (Hypothesis H1) verification."""
from __future__ import annotations

from typing import Dict, List, Any
import numpy as np
import pandas as pd

from cita2027.adaptation.feedback_queue import CausalFeedbackQueue, ForecastFeedbackItem
from cita2027.adaptation.metrics import interval_metrics
from cita2027.adaptation.correctors import (
    BaseCorrector,
    BaselineB0_Frozen,
    BaselineB1_IndependentEWMA,
    BaselineB2_IndependentLocScale,
    BaselineB3_FullyPooled,
    HypothesisH1_SharedStateCorrector,
)
from cita2027.adaptation.synthetic_shifts import SyntheticShiftScenario


def run_prequential_experiment(
    corrector: BaseCorrector,
    scenario_data: dict,
    alpha: float = 0.20,
) -> dict:
    """Execute prequential sequence under strict causal feedback release.

    Returns:
        dict with per-horizon summary metrics and detailed trace DataFrame.
    """
    y = scenario_data["y"]
    horizons = scenario_data["horizons"]
    base_forecasts = scenario_data["base_forecasts"]
    length = scenario_data["length"]
    max_h = max(horizons)

    corrector.reset()
    queue = CausalFeedbackQueue()
    traces: list[dict] = []

    for t in range(length):
        # 1. Causal release: feedback available at time t
        ready_items = queue.step_and_release(current_t=t)
        if ready_items:
            corrector.update(ready_items)

        # 2. At origin t, forecast all horizons h
        for h in horizons:
            target_t = t + h
            if target_t >= length:
                continue

            base_samples = base_forecasts[t][h]
            adapted_samples = corrector.adapt(origin_t=t, horizon=h, base_samples=base_samples)

            # Evaluate against ground truth (strictly for metrics, not leaked to corrector)
            y_target = float(y[target_t])
            m = interval_metrics(adapted_samples, y_target, alpha=alpha)

            trace_row = {
                "origin_t": t,
                "horizon": h,
                "target_t": target_t,
                "y_true": y_target,
                **m,
            }
            traces.append(trace_row)

            # Register in causal queue for future arrival at target_t
            item = ForecastFeedbackItem(
                origin_t=t,
                horizon=h,
                target_t=target_t,
                y_true=y_target,
                base_samples=base_samples,
                adapted_samples=adapted_samples,
            )
            queue.enqueue(item)

    df_trace = pd.DataFrame(traces)

    # Compute aggregate summary by horizon
    summary: dict[int, dict[str, float]] = {}
    for h in horizons:
        sub = df_trace[df_trace["horizon"] == h]
        summary[h] = {
            "crps_mean": float(sub["crps"].mean()),
            "mae_mean": float(sub["mae"].mean()),
            "winkler_mean": float(sub["winkler"].mean()),
            "width_mean": float(sub["width"].mean()),
            "coverage": float(sub["coverage"].mean()),
            "count": int(len(sub)),
        }

    overall_crps = float(df_trace["crps"].mean())

    return {
        "summary_by_horizon": summary,
        "overall_crps": overall_crps,
        "df_trace": df_trace,
    }


def run_full_h1_benchmark(seed: int = 2027) -> dict:
    """Run full benchmark across all 8 regimes comparing B0, B1, B2, B3 and H1."""
    horizons = [1, 4, 8, 16]
    scenario = SyntheticShiftScenario(horizons=horizons, seed=seed)

    regimes = [
        "stationary",
        "mean_shift",
        "variance_shift",
        "joint_shift",
        "shared_cross_horizon_drift",
        "horizon_specific_drift",
        "abrupt_reversal",
        "transient_outliers",
    ]

    all_results = {}
    comparison_rows = []

    for reg in regimes:
        data = scenario.generate(regime=reg, length=600)
        reg_results = {}

        models = {
            "B0_Frozen": BaselineB0_Frozen(horizons),
            "B1_IndepEWMA": BaselineB1_IndependentEWMA(horizons, alpha=0.10),
            "B2_IndepLocScale": BaselineB2_IndependentLocScale(horizons, alpha=0.10),
            "B3_FullyPooled": BaselineB3_FullyPooled(horizons, alpha=0.08),
            "H1_SharedState": HypothesisH1_SharedStateCorrector(horizons, latent_dim=2, lr_shared=0.08, lr_local=0.03),
        }

        for m_name, model in models.items():
            res = run_prequential_experiment(model, data)
            reg_results[m_name] = res

            for h in horizons:
                metrics = res["summary_by_horizon"][h]
                comparison_rows.append({
                    "regime": reg,
                    "method": m_name,
                    "horizon": h,
                    "crps": metrics["crps_mean"],
                    "mae": metrics["mae_mean"],
                    "winkler": metrics["winkler_mean"],
                    "coverage": metrics["coverage"],
                })

        all_results[reg] = reg_results

    df_comparison = pd.DataFrame(comparison_rows)
    return {
        "regime_results": all_results,
        "df_comparison": df_comparison,
    }
