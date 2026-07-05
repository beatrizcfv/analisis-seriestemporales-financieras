# Análisis de dominio de frecuencia de la serie de retorno (FFT)

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq, ifft

def power_spectrum(returns: pd.Series):
    """ Calcula el espectro de una potencia unilateral de una serie de retorno mediante FFT. """
    n = len(returns)
    values = returns.values

    spectrum = fft(values)
    freqs = fftfreq(n, d=1) # Días

    positive = freqs > 0

    return freqs[positive], np.abs(spectrum[positive]) ** 2


def low_pass_filter(price_series: pd.Series, keep_fraction: float = 0.05) -> pd.Series:
    """ Elimina el ruido de una serie de precios igualando a 0 los componentes de FFT de alta
    frecuencia.

    Mantiene el 'keep_fraction' más bajo de frecuencias y reconstruye la señal con IFFT, 
    filtrando el ruido de mercado de alta frecuencia.
    """
    clean_series = price_series.dropna()
    n = len(clean_series)
    values = clean_series.values

    spectrum = fft(values)
    freqs = fftfreq(n)

    freq_threshold = np.percentile(np.abs(freqs), keep_fraction * 100)
    mask = np.abs(freqs) <= freq_threshold

    filtered_spectrum = np.zeros_like(spectrum)
    filtered_spectrum[mask] = spectrum[mask]

    reconstructed = np.real(ifft(filtered_spectrum))
    result = pd.Series(reconstructed, index=clean_series.index)

    # Realinea con el índice origial y rellena el espacio de los bordes
    return result.reindex(price_series.index).ffill().bfill()
