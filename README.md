# Quantitative Analysis and Predictive Modeling of Financial Time Series

This is an independent project applying signal processing and machine learning techniques to financial time series, using Apple (AAPL) daily price data as a case study.

The goal is to go from raw, noisy price data to: (1) a statistically sound description of its stationarity and cyclical structure and (2) a baseline machine learning model for next-day price direction, built with methodology that avoids the most common time-series pitfalls (data leakage and use of non-stationary inputs).

## Overview

Financial price series are noisy and non-stationary, which makes them hard to model directly. This project addresses that in three stages:

1. **Statistical characterization**: transform prices into log returns and volatility, and formally test for stationarity.
2. **Signal processing**: move from the time domain to the frequency domain (Fourier Transform) to separate signal from noise, and use a time-frequency method (STFT) to see how dominant cycles shift over time.
3. **Predictive modeling**: use the engineered features (volatility, denoised trend, lagged returns) to train a binary classifier (XGBoost) that predicts whether the next day's return will be positive or negative.

## Methodology

### 1. Returns and volatility
Raw closing prices are not stationary (their mean and variance drift over time), so most statistical and ML models can't use them directly. The pipeline converts prices to **log returns**, which stabilize variance and are much closer to stationary. **Rolling volatility** (21-day standard deviation of returns) is computed as a proxy for risk/market activity.

### 2. Stationarity: Augmented Dickey-Fuller (ADF) test
The ADF test is used to formally check whether a series is stationary (p-value < 0.05 rejects the null hypothesis of a unit root). This is run on both the raw price series and the log returns, confirming that returns (not prices) are the appropriate input for the downstream analysis.

### 3. Frequency-domain analysis (FFT)
The Fast Fourier Transform decomposes the return series into its constituent frequencies, producing a power spectrum that highlights which cycle lengths carry the most energy. This spectrum drives a **low-pass filter**: high frequencies (short-term noise) are zeroed out and the signal is reconstructed with the inverse FFT, producing a denoised trend line for the price series.

### 4. Time-frequency analysis (STFT)
Market regimes change over time, so a single FFT over the whole history hides when a given cycle was active. The **Short-Time Fourier Transform** applies the FFT over a sliding window, producing a spectrogram that shows how the dominant frequencies (and therefore volatility regimes) evolve over time.

### 5. Predictive model (XGBoost)
A binary classifier is trained to predict the sign of the next day's return, using:
- Rolling volatility
- Fourier-denoised price trend
- Deviation of the raw price from that trend
- Returns lagged 1 and 2 days

**Chronological train/test split (80/20)** is used instead of a random split, as it would leak future information into training, which is invalid for time series and is one of the most common mistakes when applying ML to financial data.

### 6. Robustness checks: walk-forward validation and multiple tickers
A single chronological 80/20 split reports accuracy on exactly one test window, lucky or unlucky. `validate.py` runs **walk-forward validation**: the training window expands and the model is retrained and re-evaluated across several sequential folds, producing a distribution of out-of-sample accuracies instead of one point estimate. The same procedure is repeated **independently across several tickers** (AAPL, MSFT, SPY by default) to check whether the edge generalizes beyond a single stock or is an artifact of that specific series.

## Results

On AAPL daily data (2020–2026), the model reaches roughly **55% accuracy** on a single out-of-sample chronological test set, and a comparable range once validated with walk-forward folds. In isolation, that number looks unimpressive, but in the context of daily equity price direction it is a modest, plausible edge rather than a red flag: financial markets have a low signal-to-noise ratio and are close to (though not perfectly) efficient, so accuracy much higher than 50–55% on a simple feature set is more often a sign of a leakage bug than genuine skill. The purpose of this project is the methodology (stationarity testing, leakage-free evaluation, frequency-domain feature engineering, walk-forward robustness checks), not a claim that this model is trade-ready.

## Project structure

```
analisis-seriestemporales-financieras/
├── README.md
├── requirements.txt
├── main.py                  # runs the full pipeline end-to-end, saves plots to output/
├── validate.py              # walk-forward validation across multiple tickers (robustness check)
├── src/
│   ├── data.py              # price data download (yfinance)
│   ├── features.py          # returns, rolling volatility, ADF test
│   ├── fourier.py           # FFT power spectrum, low-pass denoising filter
│   ├── spectral.py          # STFT time-frequency analysis
│   └── model.py             # feature matrix, chronological split, walk-forward validation, XGBoost training/eval
├── notebooks/
│   └── analysis.ipynb       # interactive, step-by-step walkthrough with plots
└── output/                  # generated plots (created when main.py runs)
```

## How to run

```bash
# 1. Clone the repository
git clone https://github.com/beatrizcfv/analisis-seriestemporales-financieras.git
cd analisis-seriestemporales-financieras

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python main.py
```

This downloads AAPL data, runs the full analysis, prints the ADF and model evaluation results to the console, and saves all plots to `output/`.

To check the robustness of the predictive model with walk-forward validation across AAPL, MSFT and SPY:

```bash
python validate.py
```

Alternatively, open `notebooks/analysis.ipynb` to run the single-ticker pipeline interactively, cell by cell, with inline plots.

## Tech stack

Python · pandas · NumPy · yfinance · statsmodels · SciPy (FFT, STFT) · XGBoost · scikit-learn · Matplotlib · Jupyter

## Limitations and possible extensions

- The model uses a fixed, fairly small set of features; it is a methodological baseline, not a production trading strategy.
- No transaction costs, slippage, or position sizing are modeled — the accuracy figures say nothing about whether the signal would be profitable after costs.
- `validate.py` covers three tickers (AAPL, MSFT, SPY); extending it to more assets and asset classes (FX, commodities, crypto) would give a stronger generalization check.
- A natural next step is turning the STFT/Fourier-denoised trend into explicit trading rules and backtesting them with realistic costs, rather than only reporting classification accuracy.

## Disclaimer

This project is for educational and portfolio purposes only. It does not constitute financial advice, and the model's predictions should not be used to make real trading or investment decisions.
