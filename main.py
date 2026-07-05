"""
Descargas datos de precio, funciones de volatilidad/retorno, comprobaciones de
estacionariedad, aplica filtrado de ruido basado en Fourier y análisis STFT. Entrena
un modelo XGBoost para predecir la dirección del precio al día siguiente.
"""

import os 

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

from src.data import download_price_data
from src.features import add_returns_and_volatility, adf_test
from src.fourier import power_spectrum, low_pass_filter