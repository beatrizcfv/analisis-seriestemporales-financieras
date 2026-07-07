# Time-frequency analysis via Short-Time Fourier Transform (STFT).

import pandas as pd
from scipy.signal import stft

def compute_stft(returns: pd.Series, nperseg: int = 60, noverlap: int = 50):
    """ Compute the STFT of a return series.

    Market cycles are not static, so a single FFT over the whole series
    hides when a given frequency was dominant. The STFT applies the FFT
    over a sliding window, producing a spectrogram that shows how
    frequency content evolves over time.

    Returns
    f : ndarray
        Frequency bins.
    dates : list
        Calendar dates aligned to each time segment.
    power : ndarray
        Magnitude of the STFT.
    """
    series = returns.squeeze().dropna()
    values = series.values
    index = series.index

    f, t, Zxx = stft(values, fs=1.0, nperseg=nperseg, noverlap=noverlap)
    power = abs(Zxx)

    # Map STFT segment positions to real calendar dates
    dates = [index[min(int(ti), len(index) - 1)] for ti in t]

    return f, dates, power
