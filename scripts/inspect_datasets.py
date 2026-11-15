import pandas as pd
import numpy as np
from pathlib import Path

spy_path = Path(r"D:\01_AI_Research_Projects\cita2027-phase2\artifacts\processed\SPY_20261104T095117433396Z_b41a8489.csv")
btc_candidates = list(Path(r"D:\01_AI_Research_Projects\cita2027-phase2\artifacts\btcusdt_data\snapshots").glob("*.csv"))

print("=" * 60)
print("SPY DATASET INSPECTION")
print("=" * 60)
df_spy = pd.read_csv(spy_path)
print("File:", spy_path.name)
print("Shape:", df_spy.shape)
print("Columns:", df_spy.columns.tolist())
time_col_spy = [c for c in df_spy.columns if "time" in c.lower() or "date" in c.lower()][0]
print("Time col:", time_col_spy)
print("Time min/max:", df_spy[time_col_spy].iloc[0], "to", df_spy[time_col_spy].iloc[-1])
print("Splits:\n", df_spy["split"].value_counts())
print("\nLog Return stats:")
print(df_spy["log_return"].describe())

# Autocorrelation of returns and squared returns
ret_spy = df_spy["log_return"].dropna()
print("\nAutocorrelation of raw returns at lags [1, 2, 4, 8, 16, 30, 60]:")
for lag in [1, 2, 4, 8, 16, 30, 60]:
    ac = ret_spy.autocorr(lag=lag)
    print(f"  Lag {lag:>2}: {ac:+.4f}")

print("\nAutocorrelation of squared returns (Volatility Clustering) at lags [1, 2, 4, 8, 16, 30, 60]:")
sq_ret_spy = (ret_spy ** 2)
for lag in [1, 2, 4, 8, 16, 30, 60]:
    ac = sq_ret_spy.autocorr(lag=lag)
    print(f"  Lag {lag:>2}: {ac:+.4f}")

if btc_candidates:
    btc_path = btc_candidates[0]
    print("\n" + "=" * 60)
    print("BTC DATASET INSPECTION")
    print("=" * 60)
    df_btc = pd.read_csv(btc_path)
    print("File:", btc_path.name)
    print("Shape:", df_btc.shape)
    print("Columns:", df_btc.columns.tolist())
    time_col_btc = [c for c in df_btc.columns if "time" in c.lower() or "date" in c.lower()][0]
    print("Time col:", time_col_btc)
    print("Time min/max:", df_btc[time_col_btc].iloc[0], "to", df_btc[time_col_btc].iloc[-1])
    print("Splits:\n", df_btc["split"].value_counts())
    print("\nLog Return stats:")
    print(df_btc["log_return"].describe())

    ret_btc = df_btc["log_return"].dropna()
    print("\nAutocorrelation of raw returns at lags [1, 2, 4, 8, 16, 30, 60]:")
    for lag in [1, 2, 4, 8, 16, 30, 60]:
        ac = ret_btc.autocorr(lag=lag)
        print(f"  Lag {lag:>2}: {ac:+.4f}")

    print("\nAutocorrelation of squared returns (Volatility Clustering) at lags [1, 2, 4, 8, 16, 30, 60]:")
    sq_ret_btc = (ret_btc ** 2)
    for lag in [1, 2, 4, 8, 16, 30, 60]:
        ac = sq_ret_btc.autocorr(lag=lag)
        print(f"  Lag {lag:>2}: {ac:+.4f}")
