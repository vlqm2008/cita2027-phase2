"""
Data preparation script for FJCAI Benchmark replication.
Assets: SPY (^GSPC), AAPL, MSFT, GOOG, AMZN, META, and BTCUSDT.
Period: 2010-01-01 to 2024-12-31 (or latest available).
Computes: Log returns, volume, rolling statistics (20d, 60d), momentum, calendar encodings.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(r"D:\01_AI_Research_Projects\cita2027-phase2")
DATA_DIR = ROOT / "data" / "fjcai_assets"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["SPY", "AAPL", "MSFT", "GOOG", "AMZN", "META"]

def prepare_equity(ticker: str):
    out_file = DATA_DIR / f"{ticker}_daily.csv"
    if out_file.exists():
        print(f"[{ticker}] Already cached at {out_file.name}")
        return pd.read_csv(out_file)

    print(f"[{ticker}] Downloading from yfinance (2010 to 2024)...")
    try:
        raw = yf.download(ticker, start="2010-01-01", end="2024-12-31", progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            # Flatten multi-index if returned by yfinance
            raw.columns = [col[0].lower() for col in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]
        
        raw = raw.reset_index()
        date_col = "Date" if "Date" in raw.columns else "date"
        raw.rename(columns={date_col: "date"}, inplace=True)
        raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
        raw = raw.sort_values("date").reset_index(drop=True)
        
        close_col = "adj close" if "adj close" in raw.columns else "close"
        raw["close"] = raw[close_col].astype(float)
        raw["log_return"] = np.log(raw["close"] / raw["close"].shift(1))
        
        # Additional features according to Section IV-A of FJCAI paper:
        # Calendar features (sin/cos day-of-week, month)
        raw["dow"] = raw["date"].dt.dayofweek
        raw["month"] = raw["date"].dt.month
        raw["sin_dow"] = np.sin(2 * np.pi * raw["dow"] / 5.0)
        raw["cos_dow"] = np.cos(2 * np.pi * raw["dow"] / 5.0)
        raw["sin_month"] = np.sin(2 * np.pi * raw["month"] / 12.0)
        raw["cos_month"] = np.cos(2 * np.pi * raw["month"] / 12.0)
        
        # Volume
        vol_col = "volume" if "volume" in raw.columns else None
        if vol_col and vol_col in raw:
            raw["log_volume"] = np.log1p(raw[vol_col].astype(float))
        else:
            raw["log_volume"] = 0.0
            
        # Rolling statistics (20d, 60d)
        raw["rolling_mean_20"] = raw["log_return"].rolling(20).mean()
        raw["rolling_vol_20"] = raw["log_return"].rolling(20).std()
        raw["rolling_mean_60"] = raw["log_return"].rolling(60).mean()
        raw["rolling_vol_60"] = raw["log_return"].rolling(60).std()
        
        # Momentum
        raw["mom_5"] = raw["log_return"].rolling(5).sum()
        raw["mom_20"] = raw["log_return"].rolling(20).sum()
        
        raw["asset"] = ticker
        raw = raw.dropna(subset=["log_return"]).reset_index(drop=True)
        raw.to_csv(out_file, index=False)
        print(f"[{ticker}] Saved {len(raw)} rows to {out_file.name}")
        return raw
    except Exception as e:
        print(f"[{ticker}] Error downloading: {e}")
        return None

def prepare_btc():
    out_file = DATA_DIR / "BTCUSDT_daily.csv"
    if out_file.exists():
        print(f"[BTCUSDT] Already cached at {out_file.name}")
        return pd.read_csv(out_file)

    # Check local snapshot
    local_btc_snapshots = list(Path(r"D:\01_AI_Research_Projects\cita2027-phase2\artifacts\btcusdt_data\snapshots").glob("*.csv"))
    if local_btc_snapshots:
        print(f"[BTCUSDT] Loading local snapshot: {local_btc_snapshots[0].name}")
        raw = pd.read_csv(local_btc_snapshots[0])
        time_col = [c for c in raw.columns if "date" in c.lower() or "time" in c.lower()][0]
        raw.rename(columns={time_col: "date"}, inplace=True)
        raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
        raw = raw.sort_values("date").reset_index(drop=True)
        if "close" in raw.columns:
            raw["close"] = raw["close"].astype(float)
        else:
            raw["close"] = np.exp(raw["log_return"].cumsum())
        if "log_return" not in raw.columns:
            raw["log_return"] = np.log(raw["close"] / raw["close"].shift(1))
    else:
        print("[BTCUSDT] Downloading from yfinance...")
        raw = yf.download("BTC-USD", start="2017-01-01", end="2024-12-31", progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[0].lower() for col in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]
        raw = raw.reset_index()
        raw.rename(columns={"Date": "date"}, inplace=True)
        raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
        raw["close"] = raw["adj close"].astype(float) if "adj close" in raw.columns else raw["close"].astype(float)
        raw["log_return"] = np.log(raw["close"] / raw["close"].shift(1))

    raw["dow"] = raw["date"].dt.dayofweek
    raw["month"] = raw["date"].dt.month
    raw["sin_dow"] = np.sin(2 * np.pi * raw["dow"] / 7.0)
    raw["cos_dow"] = np.cos(2 * np.pi * raw["dow"] / 7.0)
    raw["sin_month"] = np.sin(2 * np.pi * raw["month"] / 12.0)
    raw["cos_month"] = np.cos(2 * np.pi * raw["month"] / 12.0)
    raw["log_volume"] = 0.0
    raw["rolling_mean_20"] = raw["log_return"].rolling(20).mean()
    raw["rolling_vol_20"] = raw["log_return"].rolling(20).std()
    raw["rolling_mean_60"] = raw["log_return"].rolling(60).mean()
    raw["rolling_vol_60"] = raw["log_return"].rolling(60).std()
    raw["mom_5"] = raw["log_return"].rolling(5).sum()
    raw["mom_20"] = raw["log_return"].rolling(20).sum()
    raw["asset"] = "BTCUSDT"
    raw = raw.dropna(subset=["log_return"]).reset_index(drop=True)
    raw.to_csv(out_file, index=False)
    print(f"[BTCUSDT] Saved {len(raw)} rows to {out_file.name}")
    return raw

def main():
    print("Preparing all FJCAI assets...")
    for t in TICKERS:
        prepare_equity(t)
    prepare_btc()
    print("All assets prepared successfully!")

if __name__ == "__main__":
    main()
