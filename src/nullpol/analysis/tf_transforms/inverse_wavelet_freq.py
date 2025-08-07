"""functions for computing the inverse wavelet transforms"""

from __future__ import annotations

import numpy as np
from numba import njit


def inverse_wavelet_freq_helper_fast(wave_in: np.ndarray, phif: np.ndarray, Nf: int, Nt: int) -> np.ndarray:
    """Helper for fast inverse wavelet transform in frequency domain.

    Args:
        wave_in (numpy.ndarray): 2D numpy array of input data in wavelet domain.
        phif (numpy.ndarray): 1D numpy array representing the wavelet.
        Nf (int): Number of frequency bins.
        Nt (int): Number of time bins.

    Returns:
        numpy.ndarray: 1D complex numpy array result.
    """
    ND = Nf * Nt

    prefactor2s = np.zeros(Nt, np.complex128)
    res = np.zeros(ND // 2 + 1, dtype=np.complex128)

    for m in range(0, Nf + 1):
        _pack_wave_inverse(m, Nt, Nf, prefactor2s, wave_in)
        # with numba.objmode(fft_prefactor2s="complex128[:]"):
        fft_prefactor2s = np.fft.fft(prefactor2s)
        _unpack_wave_inverse(m, Nt, Nf, phif, fft_prefactor2s, res)

    return res


@njit
def _unpack_wave_inverse(
    m: int, Nt: int, Nf: int, phif: np.ndarray, fft_prefactor2s: np.ndarray, res: np.ndarray
) -> None:
    """Helper for unpacking results of frequency domain inverse transform.

    Args:
        m (int): Frequency index.
        Nt (int): Number of time bins.
        Nf (int): Number of frequency bins.
        phif (numpy.ndarray): 1D numpy array wavelet.
        fft_prefactor2s (numpy.ndarray): 1D numpy array prefactors of FFT.
        res (numpy.ndarray): 1D numpy array for results.
    """
    if m == 0 or m == Nf:
        for i_ind in range(0, Nt // 2):
            i = np.abs(m * Nt // 2 - i_ind)  # i_off+i_min2
            ind3 = (2 * i) % Nt
            res[i] += fft_prefactor2s[ind3] * phif[i_ind]
        if m == Nf:
            i_ind = Nt // 2
            i = np.abs(m * Nt // 2 - i_ind)  # i_off+i_min2
            ind3 = 0
            res[i] += fft_prefactor2s[ind3] * phif[i_ind]
    else:
        ind31 = (Nt // 2 * m) % Nt
        ind32 = (Nt // 2 * m) % Nt
        for i_ind in range(0, Nt // 2):
            i1 = Nt // 2 * m - i_ind
            i2 = Nt // 2 * m + i_ind
            # assert ind31 == i1%Nt
            # assert ind32 == i2%Nt
            res[i1] += fft_prefactor2s[ind31] * phif[i_ind]
            res[i2] += fft_prefactor2s[ind32] * phif[i_ind]
            ind31 -= 1
            ind32 += 1
            if ind31 < 0:
                ind31 = Nt - 1
            if ind32 == Nt:
                ind32 = 0

        res[Nt // 2 * m] = fft_prefactor2s[(Nt // 2 * m) % Nt] * phif[0]


# @njit()
# def unpack_wave_inverse(m,Nt,Nf,phif,fft_prefactor2s,res):
#    """helper for unpacking results of frequency domain inverse transform"""
#    ND = Nt*Nf
#    i_min2 = min(max(Nt//2*(m-1),0),ND//2+1)
#    i_max2 = min(max(Nt//2*(m+1),0),ND//2+1)
#    for i in range(i_min2,i_max2):
#        i_ind = np.abs(i-Nt//2*m)
#        if i_ind>Nt//2:
#            continue
#        if m==0:
#            res[i] += fft_prefactor2s[(2*i)%Nt]*phif[i_ind]
#        elif m==Nf:
#            res[i] += fft_prefactor2s[(2*i)%Nt]*phif[i_ind]
#        else:
#            res[i] += fft_prefactor2s[i%Nt]*phif[i_ind]


@njit
def _pack_wave_inverse(m: int, Nt: int, Nf: int, prefactor2s: np.ndarray, wave_in: np.ndarray) -> None:
    """Helper for fast frequency domain inverse transform to preare for Fourier transform.

    Args:
        m (int): Frequency index.
        Nt (int): Number of time bins.
        Nf (int): Number of frequency bins.
        prefactor2s (numpy.ndarray): 1D complex numpy array prefactors.
        wave_in (numpy.ndarray): 2D numpy array input data in wavelet domain.
    """
    if m == 0:
        for n in range(0, Nt):
            prefactor2s[n] = 1 / np.sqrt(2) * wave_in[(2 * n) % Nt, 0]
    elif m == Nf:
        for n in range(0, Nt):
            prefactor2s[n] = 1 / np.sqrt(2) * wave_in[(2 * n) % Nt + 1, 0]
    else:
        for n in range(0, Nt):
            val = wave_in[n, m]
            if (n + m) % 2:
                mult2 = -1j
            else:
                mult2 = 1

            prefactor2s[n] = mult2 * val
