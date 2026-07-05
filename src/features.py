# Retorno, volatilidad y características estacionarias

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def add_returns_and_volatility(df: pd.DataFrame, price_col: str = 'Close', window: int = 21) -> pd.DataFrame:
    """ Añade rendimientos logarítmicos y volatilidad continua a un DataFrame de precios.

    Se utilizan retornos logarítmicos en lugar de precios brutos porque se estabilizan
    varianza y generalmente están más cerca de la estacionaria, que es una
    requisito para la mayoría de los modelos de series temporales y ML.
    """
    df = df.copy()
    df['Returns'] = np.log(df[price_col] / df[price_col].shift(1))
    df['Volatility'] = df['Returns'].rolling(window=window).std()
    df.dropna(inplace=True)
    
    return df


def adf_test(series: pd.Series, label: str = "") -> float:
    """ Realiza el test de Dickey-Fuller Aumentado y devuelve el p-valor.

    p-valor < 0.05 nos ayuda a rechazar la hipótesis nula de la raíz de la unidad,
    es decir, podemos tratar la serie como estacionaria
    """
    result = adfuller(series.dropna())
    p_value = result[1]
    print(f"ADF test [{label}], p-value: {p_value:.4f}")

    return p_value