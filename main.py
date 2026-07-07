"""
End-to-end pipeline.

Downloads price data, engineers return/volatility features, checks
stationarity, applies Fourier-based noise filtering and STFT analysis,
and trains an XGBoost model to predict next-day price direction.
"""

import os 

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

from src.data import download_price_data
from src.features import add_returns_and_volatility, adf_test
from src.fourier import power_spectrum, low_pass_filter
from src.model import feature_matrix, chronological_split, evaluate_model, train_model
from src.spectral import compute_stft

TICKER = 'APPL'
START_DATE = '2020-01-01'
END_DATE = '2026-01-01'
OUTPUT_DIR = 'output'


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Data acquisition and feature engineering
    df = download_price_data(TICKER, START_DATE, END_DATE)
    df = add_returns_and_volatility(df)

    # 2. Stationarity check (raw prices vs. log returns)
    adf_test(df["Close"], label=f"Close price ({TICKER})")
    adf_test(df["Returns"], label=f"Log returns ({TICKER})")

    # 3. Autocorrelation of returns
    plot_acf(df["Returns"], lags=40, zero=False)
    plt.title(f"Return Autocorrelation - {TICKER}")
    plt.savefig(f"{OUTPUT_DIR}/autocorrelation.png")
    plt.close()

    # 4. Frequency spectrum of returns
    freqs, power = power_spectrum(df["Returns"])
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, power)
    plt.title(f"Power Spectrum of Returns - {TICKER}")
    plt.xlabel("Frequency (1/day)")
    plt.ylabel("Power")
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/power_spectrum.png")
    plt.close()

    # 5. Fourier low-pass filtering (denoising the price series)
    df["Close_Clean"] = low_pass_filter(df["Close"].iloc[:, 0], keep_fraction=0.05)
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df["Close"], label="Raw Price", color="lightgray", alpha=0.7)
    plt.plot(df.index, df["Close_Clean"], label="Fourier Low-Pass Filter (5%)", color="blue", linewidth=2.5)
    plt.title("Noise Filtering via Fourier Transform")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.savefig(f"{OUTPUT_DIR}/fourier_filter.png")
    plt.close()

    # 6. Time-frequency analysis (STFT spectrogram)
    f, dates, power_stft = compute_stft(df["Returns"])
    plt.figure(figsize=(14, 6))
    plt.pcolormesh(dates, f, power_stft, shading="gouraud", cmap="viridis")
    plt.title(f"STFT Spectrogram - {TICKER}")
    plt.xlabel("Date")
    plt.ylabel("Frequency (1/day)")
    plt.colorbar(label="Cycle Intensity")
    plt.savefig(f"{OUTPUT_DIR}/stft_spectrogram.png")
    plt.close()

    # 7. Predictive model (chronological split to avoid data leakage)
    features = feature_matrix(df)
    X_train, X_test, y_train, y_test = chronological_split(features)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    print(f"\nPlots saved to '{OUTPUT_DIR}/'")


if __name__ == '__main__':
    main()
