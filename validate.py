""" Robustness check: walk-forward validation across multiple tickers.

A single train/test split on a single ticker is not enough evidence
that the model has a genuine edge. This script re-runs the feature
pipeline and walk-forward validation independently for each ticker in
'TICKERS', so the reported accuracy can be judged as a range across
folds and assets rather than a single, possibly lucky, number.
"""


from src.data import download_price_data
from src.features import add_returns_and_volatility
from src.fourier import low_pass_filter
from src.model import feature_matrix, walk_forward_validate

TICKERS  = ['AAPL', 'MSFT', 'SPY']
START_DATE = '2020-01-01'
END_DATE = '2026-01-01'
N_SPLITS = 5

def run_ticker(ticker: str) -> dict:
    df = download_price_data(ticker, START_DATE, END_DATE)
    df = add_returns_and_volatility(df)
    df['Close_Clean'] = low_pass_filter(df['Close'].iloc[:, 0], keep_fraction=0.05)

    features = feature_matrix(df)
    print(f"\n=== {ticker} ===")

    return walk_forward_validate(features, n_splits=N_SPLITS)


def main():
    summary = {}
    for ticker in TICKERS:
        summary[ticker] = run_ticker(ticker)

    print("\nSummary (mean accuracy +/- std across folds):")
    for ticker, result in summary.items():
        print(f"  {ticker}: {result['mean_accuracy']:.4f}   +/- {result['std_accuracy']:.4f}")


if __name__ == "__main__":
    main()
