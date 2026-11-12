"""Export prequential trajectory data for interactive live visualization."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from cita2027.adaptation.correctors import (
    BaselineB0_Frozen,
    BaselineB1_IndependentEWMA,
    HypothesisH1_SharedStateCorrector,
)
from cita2027.adaptation.hypothesis_h2 import HypothesisH2_AgeWeightedCorrector
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector
from cita2027.adaptation.synthetic_shifts import SyntheticShiftScenario
from cita2027.adaptation.feedback_queue import CausalFeedbackQueue, ForecastFeedbackItem
from cita2027.adaptation.metrics import interval_metrics, crps_empirical_samples
from cita2027.adaptation.real_data_benchmark import generate_or_load_real_forecasts


def generate_trace_for_scenario(scenario_name: str, scenario_data: dict, horizons: list[int] = [1, 4, 8, 16]) -> dict:
    y = scenario_data["y"]
    base_fc = scenario_data["base_forecasts"]
    length = min(scenario_data["length"], 120)

    models = {
        "B0_Frozen": BaselineB0_Frozen(horizons),
        "B1_IndepEWMA": BaselineB1_IndependentEWMA(horizons, alpha=0.10),
        "H1_SharedState": HypothesisH1_SharedStateCorrector(horizons, latent_dim=2),
        "H3_Unified": HypothesisH3_OverlapAwareCorrector(horizons, latent_dim=2, gamma_age=0.12, overlap_discount_factor=0.7),
    }

    queues = {m: CausalFeedbackQueue() for m in models}
    step_records = []

    for t in range(length):
        step_entry = {
            "t": t,
            "y_true": float(y[t]),
            "matured_horizons": [],
            "models": {},
        }

        # Check feedback
        for m_name, model in models.items():
            ready = queues[m_name].step_and_release(current_t=t)
            if ready:
                model.update(ready)
                if m_name == "H3_Unified":
                    step_entry["matured_horizons"] = [it.horizon for it in ready]

            # Record model state
            state_vec = [0.0, 0.0]
            if isinstance(model, HypothesisH3_OverlapAwareCorrector):
                state_vec = [float(model.shared_state[0]), float(model.shared_state[1])]
            elif isinstance(model, HypothesisH1_SharedStateCorrector):
                state_vec = [float(model.shared_state[0]), float(model.shared_state[1])]

            step_entry["models"][m_name] = {
                "state": state_vec,
                "horizons": {},
            }

            for h in horizons:
                target_t = t + h
                if target_t < len(y):
                    b_samp = base_fc[t][h]
                    a_samp = model.adapt(origin_t=t, horizon=h, base_samples=b_samp)
                    y_targ = float(y[target_t])

                    m_metrics = interval_metrics(a_samp, y_targ, alpha=0.20)
                    step_entry["models"][m_name]["horizons"][h] = {
                        "median": float(np.median(a_samp)),
                        "q10": float(np.percentile(a_samp, 10)),
                        "q90": float(np.percentile(a_samp, 90)),
                        "crps": float(m_metrics["crps"]),
                    }

                    item = ForecastFeedbackItem(
                        origin_t=t,
                        horizon=h,
                        target_t=target_t,
                        y_true=y_targ,
                        base_samples=b_samp,
                        adapted_samples=a_samp,
                    )
                    queues[m_name].enqueue(item)

        step_records.append(step_entry)

    return {
        "scenario": scenario_name,
        "horizons": horizons,
        "steps": step_records,
    }


def generate_trace_for_real(dataset_name: str, fc_data: dict, horizons: list[int] = [1, 4, 8, 16]) -> dict:
    samples = fc_data["samples"]
    targets = fc_data["targets"]
    dates = fc_data["dates"]
    n_origins = min(fc_data["N_origins"], 80)

    models = {
        "B0_Frozen": BaselineB0_Frozen(horizons),
        "B1_IndepEWMA": BaselineB1_IndependentEWMA(horizons, alpha=0.10),
        "H1_SharedState": HypothesisH1_SharedStateCorrector(horizons, latent_dim=2),
        "H3_Unified": HypothesisH3_OverlapAwareCorrector(horizons, latent_dim=2, gamma_age=0.12, overlap_discount_factor=0.7),
    }
    queues = {m: CausalFeedbackQueue() for m in models}
    step_records = []

    for t in range(n_origins):
        step_entry = {
            "t": t,
            "date": dates[t],
            "y_true": float(targets[t, 0]),  # 1-day ahead real return
            "matured_horizons": [],
            "models": {},
        }

        for m_name, model in models.items():
            ready = queues[m_name].step_and_release(current_t=t)
            if ready:
                model.update(ready)
                if m_name == "H3_Unified":
                    step_entry["matured_horizons"] = [it.horizon for it in ready]

            state_vec = [0.0, 0.0]
            if hasattr(model, "shared_state"):
                state_vec = [float(model.shared_state[0]), float(model.shared_state[1])]

            step_entry["models"][m_name] = {
                "state": state_vec,
                "horizons": {},
            }

            for h in horizons:
                target_t = t + h
                h_idx = h - 1

                b_samp = samples[t, :, h_idx]
                a_samp = model.adapt(origin_t=t, horizon=h, base_samples=b_samp)
                y_targ = float(targets[t, h_idx])

                m_metrics = interval_metrics(a_samp, y_targ, alpha=0.20)
                step_entry["models"][m_name]["horizons"][h] = {
                    "median": float(np.median(a_samp)),
                    "q10": float(np.percentile(a_samp, 10)),
                    "q90": float(np.percentile(a_samp, 90)),
                    "crps": float(m_metrics["crps"]),
                }

                if target_t < n_origins:
                    item = ForecastFeedbackItem(
                        origin_t=t,
                        horizon=h,
                        target_t=target_t,
                        y_true=y_targ,
                        base_samples=b_samp,
                        adapted_samples=a_samp,
                    )
                    queues[m_name].enqueue(item)

        step_records.append(step_entry)

    return {
        "scenario": dataset_name,
        "horizons": horizons,
        "steps": step_records,
    }


def main():
    root = Path(".").resolve()
    out_dir = root / "artifacts" / "adaptation" / "visualization"
    out_dir.mkdir(parents=True, exist_ok=True)

    horizons = [1, 4, 8, 16]
    scenario = SyntheticShiftScenario(horizons=horizons, seed=2027)

    all_scenarios = {}

    print("Generating Joint Shift visual trajectory...")
    data_joint = scenario.generate(regime="joint_shift", length=100)
    all_scenarios["Joint Shift (Mean+Var)"] = generate_trace_for_scenario("Joint Shift", data_joint, horizons)

    print("Generating Abrupt Reversal visual trajectory...")
    data_abrupt = scenario.generate(regime="abrupt_reversal", length=100)
    all_scenarios["Abrupt Reversal"] = generate_trace_for_scenario("Abrupt Reversal", data_abrupt, horizons)

    print("Generating Stationary visual trajectory...")
    data_stat = scenario.generate(regime="stationary", length=100)
    all_scenarios["Stationary Noise"] = generate_trace_for_scenario("Stationary", data_stat, horizons)

    print("Loading Real SPY visual trajectory...")
    spy_fc = generate_or_load_real_forecasts("SPY", split="validation", horizons=horizons, max_origins=80, root_dir=root)
    all_scenarios["SPY (S&P 500 Daily Returns)"] = generate_trace_for_real("SPY", spy_fc, horizons)

    print("Loading Real BTCUSDT visual trajectory...")
    btc_fc = generate_or_load_real_forecasts("BTCUSDT", split="validation", horizons=horizons, max_origins=80, root_dir=root)
    all_scenarios["BTCUSDT (Crypto Daily Returns)"] = generate_trace_for_real("BTCUSDT", btc_fc, horizons)

    out_file = out_dir / "visualization_traces.json"
    with open(out_file, "w") as f:
        json.dump(all_scenarios, f)
    print(f"[OK] Exported visualization dataset to {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
