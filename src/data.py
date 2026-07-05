# Adquisición de datos.

import pandas as pd
import yfinance as yf

def download_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    # Descarga histórica de un ticker determinado
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)
    return pd.DataFrame(data)
