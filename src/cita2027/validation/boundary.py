"""TASK-006A: target-based SPY Validation plan and causal windows.

No GPU imports. Do not call the development-only make_window for Validation:
its split guard explicitly rejects the Development->Validation boundary.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cita2027.forecasting.cache import digest, store_or_verify
from cita2027.forecasting.protocol import ForecastWindow
from cita2027.utils.provenance import sha256_file

DEV_PLAN_SHA = '39c14b72d7bd82b815757147838e45fe3df56bddf4616367139cfd5661bd089f'
PROCESSED_SHA = '4375f3b90c5dbfe1cfd1a8191a36c15c2d878993ef04b130c8de8810927103ba'
DEV_LAST_ORIGIN = 4529
DEV_LAST_TARGET = 4530
VAL_FIRST_TARGET = 4531
VAL_LAST_TARGET = 5032
VAL_COUNT = 502
SOURCE_ID = 'SPY_20261104T095117433396Z_b41a8489'


def validation_window(df: pd.DataFrame, i: int) -> ForecastWindow:
    """Predict row i+1 from observed returns [i-255, i], never y[i+1]."""
    if not 255 <= i < len(df) - 1:
        raise ValueError('Insufficient causal context or missing next-session target')
    origin, target = df.iloc[i], df.iloc[i + 1]
    if target['split'] != 'validation' or origin['split'] not in ('development', 'validation'):
        raise ValueError('Validation target or origin split violation')
    if i == VAL_FIRST_TARGET - 1 and origin['split'] != 'development':
        raise ValueError('Missing Development-to-Validation boundary bridge')
    if i >= VAL_FIRST_TARGET and origin['split'] != 'validation':
        raise ValueError('Non-validation origin after bridge')
    if (pd.Timestamp(target['previous_date']) != pd.Timestamp(origin['date']) or
            int(target['previous_row_index']) != i):
        raise ValueError('Nonadjacent trading sessions or corrupted previous-row reference')
    ctx = df.iloc[i - 255:i + 1]['log_return'].to_numpy(dtype=np.float64, copy=True)
    if ctx.shape != (256,) or not np.isfinite(ctx).all():
        raise ValueError('Missing/invalid causal historical returns')
    # Observation y[i+1] is used solely for *post-hoc scoring and state update*.
    y = float(target['log_return'])
    if not np.isfinite(y):
        raise ValueError('Nonfinite realized validation target')
    ctx.setflags(write=False)
    return ForecastWindow(i, i + 1, origin['date'].date().isoformat(),
                          target['date'].date().isoformat(), ctx, y)


def validation_indices(df: pd.DataFrame) -> list[int]:
    if not df['date'].is_monotonic_increasing or df['date'].duplicated().any():
        raise ValueError('Nonchronological calendar')
    dev = np.flatnonzero(df['split'].to_numpy() == 'development')
    val = np.flatnonzero(df['split'].to_numpy() == 'validation')
    if len(dev) < 2 or len(val) != VAL_COUNT:
        raise ValueError('Incorrect split lengths')
    if not np.array_equal(val, np.arange(VAL_FIRST_TARGET, VAL_LAST_TARGET + 1)):
        raise ValueError('Validation targets are noncontiguous or misindexed')
    if (dev[-1] != DEV_LAST_TARGET or dev[-2] != DEV_LAST_ORIGIN or
            df.iloc[VAL_FIRST_TARGET - 1]['split'] != 'development' or
            df.iloc[VAL_LAST_TARGET + 1]['split'] != 'final_test'):
        raise ValueError('Development/Validation/Final boundary drift')
    indices = [j - 1 for j in val]
    if indices != list(range(DEV_LAST_TARGET, VAL_LAST_TARGET)):
        raise ValueError('Unexpected validation forecast origin indices')
    return indices


def make_validation_plan(df: pd.DataFrame, dev_plan: dict[str, Any],
                         checkpoint_audit_sha256: str) -> dict[str, Any]:
    if (dev_plan['count'] != 1258 or dev_plan['last']['origin_index'] != DEV_LAST_ORIGIN or
            dev_plan['last']['target_index'] != DEV_LAST_TARGET or
            dev_plan['processed_sha256'] != PROCESSED_SHA):
        raise ValueError('Previously accepted Development plan mismatch')
    if checkpoint_audit_sha256 != dev_plan['checkpoint_audit_sha256']:
        raise ValueError('Checkpoint audit identity drift')
    if (dev_plan['source_snapshot_id'] != SOURCE_ID or
            dev_plan['seed'] != 2027 or dev_plan['context_length'] != 256 or
            dev_plan['horizon'] != 1 or dev_plan['samples'] != 100 or
            dev_plan['quantile_levels'] != [0.1, 0.5, 0.9]):
        raise ValueError('Locked model/forecast protocol drift')
    origins = []
    for i in validation_indices(df):
        w = validation_window(df, i)
        origins.append({
            'origin_index': i, 'target_index': i + 1,
            'origin_date': w.origin_date, 'target_date': w.target_date,
            'context_first_index': i - 255, 'context_last_index': i,
            'context_sha256': w.context_sha256, 'seed': 2027 + i,
            'role': 'boundary_bridge' if i == DEV_LAST_TARGET else 'validation_origin',
        })
    if (origins[0]['origin_date'] != '2022-12-30' or
            origins[0]['target_date'] != '2023-01-03' or
            origins[1]['origin_date'] != '2023-01-03' or
            origins[-1]['target_index'] != VAL_LAST_TARGET):
        raise ValueError('Validation boundary date drift')
    return {
        'schema_version': 'validation-target-origin-plan-v1',
        'kind': 'target_in_validation_including_boundary_bridge',
        'development_plan_sha256': DEV_PLAN_SHA,
        'processed_sha256': PROCESSED_SHA,
        'source_snapshot_id': SOURCE_ID,
        'smoke_config_sha256': dev_plan['smoke_config_sha256'],
        'checkpoint_audit_sha256': checkpoint_audit_sha256,
        'context_length': 256, 'horizon': 1, 'samples': 100,
        'quantile_levels': [0.1, 0.5, 0.9],
        'empirical_quantile_method': 'numpy_linear',
        'seed': 2027, 'seed_policy': 'global_seed_plus_origin_index',
        'replay_origins': 2, 'replay_tolerance': 1e-5,
        'target_units': 'daily_log_return_decimal',
        'count': len(origins), 'bridge_count': 1, 'target_split': 'validation',
        'first': origins[0], 'last': origins[-1], 'origins': origins,
        'calibration_state': 'causal_continuous_replay_1258_dev_then_502_validation_no_reset',
        'nominal_miscoverage': 0.2, 'calibration_window': 252,
        'base_gamma': 0.01, 'ewma_decay': 0.94,
        'shared_gamma_driver': 'return_ewma', 'gamma_driver_bounds': [0.5, 2.0],
        'primary_correction': 'expansion_only',
        'regime_rule': 'rolling_prior_252_origin_return_EWMA_sigma_quantile_0.75',
        'bootstrap_lengths': [10, 20, 40], 'bootstrap_replicates': 1000,
        'bootstrap_seed': 2027,
        'factorial_scales': ['none', 'residual_ewma', 'return_ewma'],
        'factorial_controllers': ['rolling_cqr', 'fixed_gamma_aci', 'variable_gamma_aci'],
        'contamination_free_verified': False,
        'point_in_time_adjusted_prices_verified': False,
        'scientific_clearance': False,
    }


def publish_validation_plan(root: Path, plan: dict) -> tuple[Path, str]:
    sha = digest(plan)
    path = root / 'artifacts' / 'forecast_cache' / 'validation' / 'plans' / f'{sha}.json'
    store_or_verify(path, plan)
    if sha256_file(path) != sha:
        raise ValueError('Published plan SHA does not match canonical bytes')
    return path, sha


def check_boundary_records(dev_records: list[dict], validation_records: list[dict],
                           dev_plan: dict, val_plan: dict) -> None:
    """Ensure replay can carry state through the bridge without missing a target."""
    dev_indices = [r['origin_index'] for r in dev_records]
    val_indices = [r['origin_index'] for r in validation_records]
    if (len(dev_indices) != 1258 or dev_indices != list(range(3272, 4530)) or
            len(val_indices) != VAL_COUNT or val_indices != list(range(4530, 5032))):
        raise ValueError('Missing/duplicate forecast at Development/Validation transition')
    if dev_records[-1]['target_index'] != val_records_start(val_plan):
        raise ValueError('Development final target != Validation bridge origin')
    for rec, row in zip(validation_records, val_plan['origins']):
        if (rec['origin_index'], rec['target_index'], rec['seed']) != (row['origin_index'], row['target_index'], row['seed']):
            raise ValueError('Validation record origin/target/seed mismatch')
    if validation_records[0]['origin_date'] != dev_records[-1]['target_date']:
        raise ValueError('Noncausal transition / missing bridge target')


def val_records_start(val_plan: dict) -> int:
    return val_plan['origins'][0]['origin_index']
