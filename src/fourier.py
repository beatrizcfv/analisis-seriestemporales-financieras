# Frequency-domain analysis of return series (FFT).

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq, ifft

def power_spectrum(returns: pd.Series):
    """ Compute the one-sided power spectrum of a return series via FFT. """
    n = len(returns)
    values = returns.values

    spectrum = fft(values)
    freqs = fftfreq(n, d=1) # Días

    positive = freqs > 0

    return freqs[positive], np.abs(spectrum[positive]) ** 2


def low_pass_filter(price_series: pd.Series, keep_fraction: float = 0.05) -> pd.Series:
    """ Denoise a price series by zeroing out high-frequency FFT components.

    Keeps the lowest 'keep_fraction' of frequencies and reconstructs the
    signal with the inverse FFT, filtering out high-frequency market noise.
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

    # Re-align with the original index and fill edge gaps
    return result.reindex(price_series.index).ffill().bfill()
