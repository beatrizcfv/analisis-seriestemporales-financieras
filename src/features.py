# Return, volatility and stationarity features.

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def add_returns_and_volatility(df: pd.DataFrame, price_col: str = 'Close', window: int = 21) -> pd.DataFrame:
    """ Add log returns and rolling volatility to a price DataFrame.

    Log returns are used instead of raw prices because they stabilize
    variance and are generally closer to stationary, which is a
    requirement for most time-series and ML models.
    """
    df = df.copy()
    df['Returns'] = np.log(df[price_col] / df[price_col].shift(1))
    df['Volatility'] = df['Returns'].rolling(window=window).std()
    df.dropna(inplace=True)
    
    return df


def adf_test(series: pd.Series, label: str = "") -> float:
    """ Run an Augmented Dickey-Fuller test and return the p-value.

    p-value < 0.05 lets us reject the null hypothesis of a unit root,
    i.e. the series can be treated as stationary.
    """
    result = adfuller(series.dropna())
    p_value = result[1]
    print(f"ADF test [{label}], p-value: {p_value:.4f}")

    return p_value
