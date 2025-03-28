"""helper functions for transform_time.py"""
from __future__ import annotations

import numpy as np

from .helper import get_shape_of_wavelet_transform
from .inverse_wavelet_freq_funcs import inverse_wavelet_freq_helper_fast
from .inverse_wavelet_time_funcs import inverse_wavelet_time_helper_fast
from .transform_freq_funcs import (phitilde_vec_norm,
                                   transform_wavelet_freq_helper,
                                   transform_wavelet_freq_quadrature_helper)
from .transform_time_funcs import phi_vec, transform_wavelet_time_helper


def inverse_wavelet_time(wave_in, nx=4., mult=32):
    """Fast inverse wavelet transform to time domain.

    Args:
        wave_in (2D numpy array): Wavelet domain data.
        nx (float, optional): Steepness of the filter. Defaults to 4..
        mult (int, optional): mult. Defaults to 32.

    Returns:
        1D numpy array: Time domain data.
    """
    Nt, Nf = wave_in.shape
    time_domain_length = Nt * Nf
    # make sure K isn't bigger than ND
    mult = min(mult, Nt//2)
    phi = phi_vec(Nf, nx=nx, mult=mult)/2
    output = inverse_wavelet_time_helper_fast(wave_in, phi, Nf, Nt, mult)
    return output / np.sqrt(time_domain_length)


def inverse_wavelet_freq_time(wave_in, nx=4.):
    """Inverse wavelet transform to time domain via Fourier transform
    of frequency domain.

    Args:
        wave_in (2D numpy array): Wavelet domain data.
        nx (float, optional): Steepness of the filter. Defaults to 4..

    Returns:
        1D numpy array: Time domain data.
    """
    res_f = inverse_wavelet_freq(wave_in, nx)
    return np.fft.irfft(res_f)


def inverse_wavelet_freq(
        wave_in,
        nx=4.):
    """Inverse wavelet transform to frequency domain signal.

    Args:
        wave_in (2D numpy array): Wavelet domain data.
        nx (float, optional): Steepness of the filter. Defaults to 4..

    Returns:
        1D numpy array: Frequency domain data.
    """
    Nt, Nf = wave_in.shape
    time_domain_length = Nt * Nf
    phif = phitilde_vec_norm(Nf, Nt, nx)
    output = inverse_wavelet_freq_helper_fast(wave_in, phif, Nf, Nt)
    return output / np.sqrt(time_domain_length)


def transform_wavelet_time(
        data,
        sampling_frequency,
        frequency_resolution,
        nx=4.,
        mult=32):
    """Do the wavelet transform in the time domain,
    note there can be significant leakage if mult is too small and the
    transform is only approximately exact if mult=Nt/2.

    Args:
        data (1D numpy array): Time domain data.
        sampling_frequency (float): Sampling frequency in Hz.
        frequency_resolution (float): Frequency resolution in Hz.
        nx (float, optional): Steepness of the filter. Defaults to 4..
        mult (int, optional): mult. Defaults to 32.

    Returns:
        2D numpy array: Wavelet domain data.
    """
    time_domain_length = len(data)
    duration = time_domain_length / sampling_frequency
    Nt, Nf = get_shape_of_wavelet_transform(
        duration=duration,
        sampling_frequency=sampling_frequency,
        wavelet_frequency_resolution=frequency_resolution)
    # make sure K isn't bigger than ND
    mult = min(mult, Nt//2)
    phi = phi_vec(Nf, nx, mult)
    wave = transform_wavelet_time_helper(data, Nf, Nt, phi, mult)
    return wave * np.sqrt(time_domain_length)


def transform_wavelet_freq_time(
        data,
        sampling_frequency,
        frequency_resolution,
        nx=4.):
    """Transform time domain data into wavelet domain via FFT
    and then frequency transform.

    Args:
        data (1D numpy array): Time domain data.
        sampling_frequency (float): Sampling frequency in Hz.
        frequency_resolution (float): Frequency resolution in Hz.
        nx (float, optional): Steepness of the filter. Defaults to 4..

    Returns:
        2D numpy array: Wavelet domain data.
    """
    data_fft = np.fft.rfft(data)
    return transform_wavelet_freq(
        data=data_fft,
        sampling_frequency=sampling_frequency,
        frequency_resolution=frequency_resolution,
        nx=nx)


def transform_wavelet_freq_time_quadrature(
        data,
        sampling_frequency,
        frequency_resolution,
        nx=4.):
    """Transform time domain data into wavelet quadrature domain via FFT
    and then frequency transform.

    Args:
        data (1D numpy array): Time domain data.
        sampling_frequency (float): Sampling frequency in Hz.
        frequency_resolution (float): Frequency resolution in Hz.
        nx (float, optional): Steepness of the filter. Defaults to 4..

    Returns:
        2D numpy array: Wavelet domain data.
    """
    data_fft = np.fft.rfft(data)
    return transform_wavelet_freq_quadrature(
        data=data_fft,
        sampling_frequency=sampling_frequency,
        frequency_resolution=frequency_resolution,
        nx=nx)


def transform_wavelet_freq_quadrature(
        data,
        sampling_frequency,
        frequency_resolution,
        nx=4.):
    """Do the wavelet quadrature transform using the fast wavelet domain transform.

    Args:
        data (1D numpy array): Frequency domain data.
        sampling_frequency (float): Sampling frequency in Hz.
        frequency_resolution (float): Frequency resolution in Hz.
        nx (float, optional): Steepness of the filter. Defaults to 4..

    Returns:
        2D numpy array: Wavelet domain data.
    """
    # Assume the length of the time domain data is even.
    time_domain_length = (len(data) - 1) * 2
    duration = time_domain_length / sampling_frequency
    Nt, Nf = get_shape_of_wavelet_transform(
        duration=duration,
        sampling_frequency=sampling_frequency,
        wavelet_frequency_resolution=frequency_resolution)
    phif = 2/Nf*phitilde_vec_norm(Nf, Nt, nx)
    return transform_wavelet_freq_quadrature_helper(data, Nf, Nt, phif) \
        * np.sqrt(time_domain_length)


def transform_wavelet_freq(
        data,
        sampling_frequency,
        frequency_resolution,
        nx=4.):
    """Do the wavelet transform using the fast wavelet domain transform.

    Args:
        data (1D numpy array): Frequency domain data.
        sampling_frequency (float): Sampling frequency in Hz.
        frequency_resolution (float): Frequency resolution in Hz.
        nx (float, optional): Steepness of the filter. Defaults to 4..

    Returns:
        2D numpy array: Wavelet domain data.
    """
    # Assume the length of the time domain data is even.
    time_domain_length = (len(data) - 1) * 2
    duration = time_domain_length / sampling_frequency
    Nt, Nf = get_shape_of_wavelet_transform(
        duration=duration,
        sampling_frequency=sampling_frequency,
        wavelet_frequency_resolution=frequency_resolution)
    phif = 2/Nf*phitilde_vec_norm(Nf, Nt, nx)

    return transform_wavelet_freq_helper(data, Nf, Nt, phif) \
        * np.sqrt(time_domain_length)
