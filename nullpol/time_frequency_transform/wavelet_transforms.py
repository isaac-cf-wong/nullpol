"""helper functions for transform_time.py"""
import numpy as np
from numba import njit
from .transform_freq_funcs import (phitilde_vec_norm,
                                   transform_wavelet_freq_helper,
                                   transform_wavelet_freq_quadrature_helper)
from .transform_time_funcs import (phi_vec,
                                   transform_wavelet_time_helper)
from .inverse_wavelet_freq_funcs import inverse_wavelet_freq_helper_fast
from .inverse_wavelet_time_funcs import inverse_wavelet_time_helper_fast


def inverse_wavelet_time(wave_in,Nf,Nt,nx=4.,mult=32):
    """Fast inverse wavelet transform to time domain.

    Parameters
    ----------
    wave_in: 2D numpy array
        Data in wavelet domain.
    Nf: int
        Number of frequency bins.
    Nt: int
        Number of time bins.
    nx: float, optional
        Steepness of the filter. Defaults to 4..
    mult: int, optional
        mult. Defaults to 32.

    Returns
    -------
    1D numpy array
        Data in time domain.
    """    
    mult = min(mult,Nt//2) #make sure K isn't bigger than ND
    phi = phi_vec(Nf,nx=nx,mult=mult)/2
    output = inverse_wavelet_time_helper_fast(wave_in,phi,Nf,Nt,mult)
    return output

def inverse_wavelet_freq_time(wave_in,Nf,Nt,nx=4.):
    """Inverse wavelet transform to time domain via Fourier transform of frequency domain.

    Parameters
    ----------
    wave_in: 2D numpy array
        Data in wavelet domain.
    Nf: int
        Number of frequency bins.
    Nt: int
        Number of time bins.
    nx: float, optional
        Steepness of the filter. Defaults to 4..

    Returns
    -------
    1D numpy array
        Data in time domain.
    """    
    res_f = inverse_wavelet_freq(wave_in,Nf,Nt,nx)
    return np.fft.irfft(res_f)

def inverse_wavelet_freq(wave_in,Nf,Nt,nx=4.):
    """Inverse wavelet transform to frequency domain signal.

    Parameters
    ----------
    wave_in: 2D numpy array
        Data in wavelet domain.
    Nf: int
        Number of frequency bins.
    Nt: int
        Number of time bins.
    nx: float, optional
        Steepness of the filter. Defaults to 4..

    Returns
    -------
    1D numpy array
        Data in frequency domain.
    """    
    phif = phitilde_vec_norm(Nf,Nt,nx)
    output = inverse_wavelet_freq_helper_fast(wave_in,phif,Nf,Nt)
    return output

def transform_wavelet_time(data,Nf,Nt,nx=4.,mult=32):
    """Do the wavelet transform in the time domain,
    note there can be significant leakage if mult is too small and the
    transform is only approximately exact if mult=Nt/2.

    Parameters
    ----------
    data: 1D numpy array
        Data in time domain.
    Nf: int
        Number of frequency bins.
    Nt: int
        Number of time bins.
    nx: float, optional
        Steepness of filter. Defaults to 4..
    mult: int, optional
        mult. Defaults to 32.

    Returns
    -------
    2D numpy array
        Data in wavelet domain.
    """    
    mult = min(mult,Nt//2) #make sure K isn't bigger than ND
    phi = phi_vec(Nf,nx,mult)
    wave = transform_wavelet_time_helper(data,Nf,Nt,phi,mult)

    return wave

def transform_wavelet_freq_time(data,Nf,Nt,nx=4.):
    """Transform time domain data into wavelet domain via FFT and then frequency transform.

    Parameters
    ----------
    data: 1D numpy array
        Data in time domain.
    Nf: int
        Number of frequency bins.
    Nt: int
        Number of time bins.
    nx: float, optional
        Steepness of filter. Defaults to 4..

    Returns
    -------
    2D numpy array
        Data in wavelet domain.
    """    
    data_fft = np.fft.rfft(data)

    return transform_wavelet_freq(data_fft,Nf,Nt,nx)

def transform_wavelet_freq_time_quadrature(data,Nf,Nt,nx=4.):
    """Transform time domain data into wavelet quadrature domain via FFT and then frequency transform.

    Parameters
    ----------
    data: 1D numpy array
        Data in time domain.
    Nf: int
        Number of frequency bins.
    Nt: int
        Number of time bins.
    nx: float, optional
        Steepness of filter. Defaults to 4..

    Returns
    -------
    2D numpy array
        Data in wavelet domain.
    """
    data_fft = np.fft.rfft(data)

    return transform_wavelet_freq_quadrature(data_fft,Nf,Nt,nx)

def transform_wavelet_freq_quadrature(data,Nf,Nt,nx=4.):
    """Do the wavelet quadrature transform using the fast wavelet domain transform.

    Parameters
    ----------
    data: 1D numpy array
        Data in frequency domain.
    Nf: int
        Number of frequency bins.
    Nt: int
        Number of time bins.
    nx: float, optional
        Steepness of filter. Defaults to 4..

    Returns
    -------
    2D numpy array
        Data in wavelet domain.
    """    
    phif = 2/Nf*phitilde_vec_norm(Nf,Nt,nx)
    return transform_wavelet_freq_quadrature_helper(data,Nf,Nt,phif)

def transform_wavelet_freq(data,Nf,Nt,nx=4.):
    """Do the wavelet transform using the fast wavelet domain transform.

    Parameters
    ----------
    data: 1D numpy array
        Data in frequency domain.
    Nf: int
        Number of frequency bins.
    Nt: int
        Number of time bins.
    nx: float, optional
        Steepness of filter. Defaults to 4..

    Returns
    -------
    2D numpy array
        Data in wavelet domain.
    """    
    phif = 2/Nf*phitilde_vec_norm(Nf,Nt,nx)
    return transform_wavelet_freq_helper(data,Nf,Nt,phif)