# Análisis de tiempo/frecuencia mediante Transformada de Fourier de corto tiempo (STFT)

import pandas as pd
from scipy.signal import stft

def compute_stft(returns: pd.Series, nperseg: int = 60, noverlap: int = 50):
    """ Calcula la STFT de una serie de retorno.

    Los ciclos de mercado no son estáticos, por lo que existe una FFT para toda la
    serie oculta cuando una frecuencia dada es dominante. STFT aplica FFT sobre
    una ventana corredera, produciendo un espectrograma que muestra cómo el contenido 
    de frecuencia evoluciona con el tiempo. 
    
    Devuelve
    f : ndarray 
        Contenedores de frecuencia. 
    fechas: lista 
        Fechas del calendario alineadas a cada segmento de tiempo. 
    potencia: ndarray 
        Magnitud del STFT.
    """
    series = returns.squeeze().dropna()
    values = series.values
    index = series.index

    f, t, Zxx = stft(values, fs=1.0, nperseg=nperseg, noverlap=noverlap)
    power = abs(Zxx)

    # Mapea las posiciones de los segmentos de STFT a fechas reales
    dates = [index[min(int(ti), len(index) - 1)] for ti in t]

    return f, dates, power