"""Visualization script for the base paper (SpectralTFT & SpectralTFT-U).

Loads the exact processed S&P 500 dataset from the original paper,
analyzes the market regimes (2020 COVID shock, 2021 Retail momentum, 2022 Inflation tightening),
computes rolling volatility, cumulative return path, and Fourier frequency spectrum,
and saves a publication-quality visualization dashboard.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# Set styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.titlesize'] = 14

def main():
    root = Path(r"D:\01_AI_Research_Projects\cita2027-phase2")
    data_path = root / "artifacts" / "processed" / "SPY_20261104T095117433396Z_b41a8489.csv"
    output_dir = root / "paper_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_img = output_dir / "base_paper_data_visualization.png"

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Filter to 2010-2024 (the exact span of SpectralTFT / SpectralTFT-U)
    df_sub = df[(df['date'] >= '2010-01-01') & (df['date'] <= '2024-12-31')].copy().reset_index(drop=True)
    df_sub['cum_return'] = df_sub['log_return'].fillna(0).cumsum()
    df_sub['rolling_vol_30d'] = df_sub['log_return'].rolling(window=30).std() * np.sqrt(252)

    # Compute Fourier Transform of Cumulative Path (Core mechanism of SpectralTFT)
    valid_returns = df_sub['log_return'].dropna().values
    n = len(valid_returns)
    cumsum_path = np.cumsum(valid_returns)
    hann_window = np.hanning(n)
    windowed_path = cumsum_path * hann_window
    fft_vals = np.fft.rfft(windowed_path)
    fft_freq = np.fft.rfftfreq(n, d=1.0)
    power_spectrum = np.abs(fft_vals) ** 2 / n

    # Create 4-panel comprehensive dashboard
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=False, gridspec_kw={'height_ratios': [1.3, 1, 1, 1]})
    fig.suptitle("SPECTRAL-TFT / SPECTRAL-TFT-U: DATASET & SPECTRAL ANALYSIS\nS&P 500 Daily Dynamics Across Multi-Horizon Regimes (2010 - 2024)", fontweight='bold', y=0.98)

    # Panel 1: Price Trajectory & Market Regimes
    ax1 = axes[0]
    ax1.plot(df_sub['date'], df_sub['close'], color='#1f77b4', lw=1.5, label='SPY Close Price (USD)')
    ax1.set_ylabel("Price (USD)", fontweight='bold')
    ax1.set_title("Panel A: Asset Trajectory & Evaluation Regimes (Training: 2010-2018, Validation: 2019, Zero-Shot Test: 2020-2022)", loc='left', fontweight='bold')
    
    # Shade regimes
    ax1.axvspan(pd.to_datetime('2010-01-01'), pd.to_datetime('2018-12-31'), color='gray', alpha=0.12, label='In-Sample Training (2010-2018)')
    ax1.axvspan(pd.to_datetime('2019-01-01'), pd.to_datetime('2019-12-31'), color='gold', alpha=0.2, label='Validation (2019)')
    ax1.axvspan(pd.to_datetime('2020-01-01'), pd.to_datetime('2020-12-31'), color='crimson', alpha=0.18, label='Regime 1: COVID Shock (2020)')
    ax1.axvspan(pd.to_datetime('2021-01-01'), pd.to_datetime('2021-12-31'), color='forestgreen', alpha=0.18, label='Regime 2: Retail Momentum (2021)')
    ax1.axvspan(pd.to_datetime('2022-01-01'), pd.to_datetime('2022-12-31'), color='darkorange', alpha=0.18, label='Regime 3: Inflation Tightening (2022)')
    ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Panel 2: Daily Log Returns (Target variable)
    ax2 = axes[1]
    ax2.plot(df_sub['date'], df_sub['log_return'], color='#2ca02c', lw=0.7, alpha=0.85, label='Daily Log Return r_t = ln(P_t / P_t-1)')
    ax2.axhline(0, color='black', lw=0.8, ls='--')
    ax2.set_ylabel("Log Return", fontweight='bold')
    ax2.set_title("Panel B: Daily Log Returns & Extreme Volatility Spikes (Target Variable)", loc='left', fontweight='bold')
    ax2.set_ylim(-0.15, 0.12)
    ax2.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9)
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Panel 3: Rolling 30-Day Annualized Volatility & Cumulative Path
    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    l1 = ax3.plot(df_sub['date'], df_sub['rolling_vol_30d'] * 100, color='purple', lw=1.2, label='30-Day Ann. Volatility (%)')
    l2 = ax3_twin.plot(df_sub['date'], df_sub['cum_return'], color='navy', lw=1.5, ls='-', label='Cumulative Path s_t = sum(r_t)')
    ax3.set_ylabel("Annualized Volatility (%)", color='purple', fontweight='bold')
    ax3_twin.set_ylabel("Cumulative Log Return", color='navy', fontweight='bold')
    ax3.set_title("Panel C: Volatility Dynamics & Cumulative Return Path (Regularization Target in SpectralTFT)", loc='left', fontweight='bold')
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Panel 4: Fourier Power Spectrum of Cumulative Trajectory
    ax4 = axes[3]
    low_freq_cutoff = 0.05  # cycles per day (~20 day period)
    low_mask = fft_freq <= low_freq_cutoff
    ax4.plot(fft_freq[1:], power_spectrum[1:], color='#333333', lw=1.0, label='Full Magnitude Spectrum |F(s)|^2')
    ax4.fill_between(fft_freq[1:], 0, power_spectrum[1:], where=low_mask[1:], color='#1f77b4', alpha=0.35, label='Low-Frequency Band (Preserved by L_med)')
    ax4.fill_between(fft_freq[1:], 0, power_spectrum[1:], where=~low_mask[1:], color='crimson', alpha=0.15, label='High-Frequency Noise Band')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel("Frequency (Cycles / Day, Log Scale)", fontweight='bold')
    ax4.set_ylabel("Power Spectral Density (Log Scale)", fontweight='bold')
    ax4.set_title("Panel D: Frequency-Domain Decomposition (Fourier Power Spectrum & Low-Pass Filter Boundary)", loc='left', fontweight='bold')
    ax4.axvline(low_freq_cutoff, color='red', ls='--', lw=1.2, label=f'Cutoff f = {low_freq_cutoff} (T = 20 days)')
    ax4.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_img, dpi=300)
    plt.close(fig)
    print(f"Successfully generated and saved figure to: {out_img}")

if __name__ == "__main__":
    main()
