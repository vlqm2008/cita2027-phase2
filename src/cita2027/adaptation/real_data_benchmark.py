"""Real-World Multi-Horizon Adaptation Benchmark.

Evaluates frozen foundation models (Chronos-T5-small) under strict asynchronous
cross-horizon feedback across real market series (SPY and BTCUSDT).
Optimized for ultra-low resource footprint (CPU-only, low priority, < 400MB RAM)
so it does not interfere with background workloads or gaming.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import psutil
import torch

from cita2027.adaptation.correctors import (
    BaseCorrector,
    BaselineB0_Frozen,
    BaselineB1_IndependentEWMA,
    BaselineB2_IndependentLocScale,
    BaselineB3_FullyPooled,
    HypothesisH1_SharedStateCorrector,
)
from cita2027.adaptation.hypothesis_h2 import HypothesisH2_AgeWeightedCorrector
from cita2027.adaptation.hypothesis_h3 import HypothesisH3_OverlapAwareCorrector
from cita2027.adaptation.feedback_queue import CausalFeedbackQueue, ForecastFeedbackItem
from cita2027.adaptation.metrics import interval_metrics


def set_low_process_priority() -> None:
    """Set process priority to IDLE or BELOW_NORMAL to yield CPU to games/apps."""
    try:
        p = psutil.Process()
        if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def find_data_path(dataset_name: str, root_dir: Optional[Path] = None) -> Path:
    """Locate the processed data CSV for SPY or BTCUSDT."""
    if root_dir is None:
        root_dir = Path(".")
    root_dir = root_dir.resolve()

    if dataset_name.upper() == "SPY":
        candidates = list((root_dir / "artifacts" / "processed").glob("SPY_*.csv"))
        if not candidates:
            raise FileNotFoundError("Could not find SPY dataset in artifacts/processed/")
        return candidates[0]
    elif dataset_name.upper() in ("BTC", "BTCUSDT"):
        candidates = list((root_dir / "artifacts" / "btcusdt_data" / "snapshots").glob("*.csv"))
        if not candidates:
            raise FileNotFoundError("Could not find BTCUSDT dataset in artifacts/btcusdt_data/snapshots/")
        return candidates[0]
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


def load_market_data(
    dataset_name: str,
    split: str = "validation",
    context_length: int = 256,
    root_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load full dataset and extract evaluation slice with context."""
    path = find_data_path(dataset_name, root_dir=root_dir)
    df = pd.read_csv(path)

    df["log_return"] = pd.to_numeric(df["log_return"], errors="coerce")
    eval_df = df[df["split"] == split].copy()
    if len(eval_df) == 0:
        raise ValueError(f"No rows found with split='{split}' in {path}")

    start_idx = eval_df.index[0]
    if start_idx < context_length:
        raise ValueError(
            f"Insufficient historical context: split starts at index {start_idx}, but context_length is {context_length}"
        )

    full_returns = df["log_return"].to_numpy(dtype=np.float32)
    return df, eval_df, full_returns


def generate_or_load_real_forecasts(
    dataset_name: str,
    split: str = "validation",
    horizons: List[int] = [1, 4, 8, 16],
    context_length: int = 256,
    num_samples: int = 50,
    max_origins: int = 80,
    device: str = "cpu",
    num_threads: int = 2,
    cache_dir: Optional[Path] = None,
    root_dir: Optional[Path] = None,
    force_regenerate: bool = False,
) -> Dict[str, Any]:
    """Generate multi-horizon forecasts with Chronos-T5-small or load from cache.

    Runs strictly on CPU with limited threads and yields priority.
    """
    set_low_process_priority()
    torch.set_num_threads(num_threads)

    if root_dir is None:
        root_dir = Path(".")
    root_dir = root_dir.resolve()

    if cache_dir is None:
        cache_dir = root_dir / "artifacts" / "adaptation" / "real_data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / f"chronos_{dataset_name.lower()}_{split}_n{max_origins}_s{num_samples}.npz"

    if cache_file.exists() and not force_regenerate:
        print(f"[CACHE] Loading cached forecasts from {cache_file}...")
        npz = np.load(cache_file, allow_pickle=True)
        return {
            "samples": npz["samples"],
            "targets": npz["targets"],
            "dates": list(npz["dates"]),
            "horizons": horizons,
            "N_origins": int(npz["samples"].shape[0]),
        }

    print(f"[MODEL] Generating Chronos forecasts for {dataset_name} ({split}, n={max_origins}, CPU threads={num_threads})...")
    df, eval_df, full_returns = load_market_data(
        dataset_name, split=split, context_length=context_length, root_dir=root_dir
    )

    max_h = max(horizons)
    eval_indices = eval_df.index.to_numpy()
    valid_origins = [idx for idx in eval_indices if idx + max_h < len(df)]
    if len(valid_origins) > max_origins:
        # Take a representative contiguous slice
        valid_origins = valid_origins[:max_origins]
    n_origins = len(valid_origins)

    from chronos import ChronosPipeline
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map=device,
        dtype=torch.float32,
    )

    all_samples = []
    all_targets = []
    origin_dates = []

    # Process 1 origin at a time to keep RAM minimal (< 300MB)
    for i, idx in enumerate(valid_origins):
        ctx = [torch.tensor(full_returns[idx - context_length:idx], dtype=torch.float32)]

        with torch.inference_mode():
            fc = pipeline.predict(ctx, prediction_length=max_h, num_samples=num_samples)
            # shape (1, num_samples, max_h)
            fc_np = fc[0].cpu().numpy()

        target_slice = full_returns[idx + 1:idx + max_h + 1]
        all_samples.append(fc_np)
        all_targets.append(target_slice)
        origin_dates.append(str(df.loc[idx, "date"]))

        if (i + 1) % 20 == 0 or (i + 1) == n_origins:
            print(f"  Processed {i + 1}/{n_origins} origins...")

    del pipeline
    gc.collect()

    samples_arr = np.array(all_samples, dtype=np.float32)  # (N, num_samples, max_h)
    targets_arr = np.array(all_targets, dtype=np.float32)  # (N, max_h)

    np.savez_compressed(
        cache_file,
        samples=samples_arr,
        targets=targets_arr,
        dates=origin_dates,
    )
    print(f"[CACHE] Saved {n_origins} multi-horizon forecasts to {cache_file}")

    return {
        "samples": samples_arr,
        "targets": targets_arr,
        "dates": origin_dates,
        "horizons": horizons,
        "N_origins": n_origins,
    }


def run_real_prequential_experiment(
    corrector: BaseCorrector,
    forecast_data: Dict[str, Any],
    alpha: float = 0.20,
) -> Dict[str, Any]:
    """Execute prequential sequence under strict causal feedback release on real market data.

    Returns:
        dict with per-horizon summary metrics and trace DataFrame.
    """
    samples = forecast_data["samples"]    # (N, num_samples, max_h)
    targets = forecast_data["targets"]    # (N, max_h)
    horizons = forecast_data["horizons"]
    dates = forecast_data["dates"]
    n_origins = forecast_data["N_origins"]

    corrector.reset()
    queue = CausalFeedbackQueue()
    traces: List[Dict[str, Any]] = []

    for t in range(n_origins):
        # 1. Causal release: feedback available at step t
        ready_items = queue.step_and_release(current_t=t)
        if ready_items:
            corrector.update(ready_items)

        # 2. At origin t, forecast all requested horizons
        for h in horizons:
            target_t = t + h
            h_idx = h - 1

            base_samples = samples[t, :, h_idx]
            adapted_samples = corrector.adapt(origin_t=t, horizon=h, base_samples=base_samples)

            y_target = float(targets[t, h_idx])
            m = interval_metrics(adapted_samples, y_target, alpha=alpha)

            trace_row = {
                "origin_t": t,
                "origin_date": dates[t],
                "horizon": h,
                "target_t": target_t,
                "y_true": y_target,
                **m,
            }
            traces.append(trace_row)

            # Enqueue for future causal availability at target_t
            if target_t < n_origins:
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

    summary_by_horizon: Dict[int, Dict[str, float]] = {}
    for h in horizons:
        sub = df_trace[df_trace["horizon"] == h]
        summary_by_horizon[h] = {
            "crps_mean": float(sub["crps"].mean()),
            "mae_mean": float(sub["mae"].mean()),
            "winkler_mean": float(sub["winkler"].mean()),
            "width_mean": float(sub["width"].mean()),
            "coverage": float(sub["coverage"].mean()),
            "count": int(len(sub)),
        }

    overall_crps = float(df_trace["crps"].mean())

    return {
        "summary_by_horizon": summary_by_horizon,
        "overall_crps": overall_crps,
        "df_trace": df_trace,
    }
