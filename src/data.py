# Data acquisition utilities.

import pandas as pd
import yfinance as yf


def download_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download historical OHLCV data for a given ticker from Yahoo Finance.

    ticker : str
        Ticker symbol, e.g. "AAPL".
    start, end : str
        Date range in "YYYY-MM-DD" format.
    """
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)
    return pd.DataFrame(data)
