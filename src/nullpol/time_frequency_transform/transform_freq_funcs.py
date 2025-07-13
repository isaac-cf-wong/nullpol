"""helper functions for transform_freq"""
from __future__ import annotations

import numpy as np
import scipy.special
from numba import njit


def phitilde_vec(om,Nf,nx=4.):
    """Compute phitilde, om i array, nx is filter steepness, defaults to 4.

    Args:
        om (numpy.ndarray): 1D numpy array of angular frequencies.
        Nf (int): Number of frequency bins.
        nx (float, optional): Filter steepness. Defaults to 4.

    Returns:
        numpy.ndarray: 1D numpy array z.
    """
    OM = np.pi  #Nyquist angular frequency
    DOM = OM/Nf #2 pi times DF
    insDOM = 1./np.sqrt(DOM)
    B = OM/(2*Nf)
    A = (DOM-B)/2
    z = np.zeros(om.size)

    mask = (np.abs(om)>= A)&(np.abs(om)<A+B)

    x = (np.abs(om[mask])-A)/B
    y = scipy.special.betainc(nx,nx, x)
    z[mask] = insDOM*np.cos(np.pi/2.*y)

    z[np.abs(om)<A] = insDOM
    return z

def phitilde_vec_norm(Nf,Nt,nx):
    """Normalize phitilde as needed for inverse frequency domain transform.

    Args:
        Nf (int): Number of frequency bins.
        Nt (int): Number of time bins.
        nx (float): Filter steepness.

    Returns:
        numpy.ndarray: 1D numpy array phif.
    """
    ND = Nf*Nt
    oms = 2*np.pi/ND*np.arange(0,Nt//2+1)
    phif = phitilde_vec(oms,Nf,nx)
    #nrm should be 1
    nrm = np.sqrt((2*np.sum(phif[1:]**2)+phif[0]**2)*2*np.pi/ND)
    nrm /= np.pi**(3/2)/np.pi
    phif /= nrm
    return phif

@njit
def tukey(data,alpha,N):
    """Apply Tukey window function to data.

    Args:
        data (numpy.ndarray): 1D numpy array data.
        alpha (float): Rolling parameter.
        N (int): Length of data.

    Returns:
        numpy.ndarray: 1D numpy array data with Tukey window function applied.
    """
    imin = np.int64(alpha*(N-1)/2)
    imax = np.int64((N-1)*(1-alpha/2))
    Nwin = N-imax

    for i in range(0,N):
        f_mult = 1.0
        if i<imin:
            f_mult = 0.5*(1.+np.cos(np.pi*(i/imin-1.)))
        if i>imax:
            f_mult = 0.5*(1.+np.cos(np.pi/Nwin*(i-imax)))
        data[i] *= f_mult

def transform_wavelet_freq_helper(data,Nf,Nt,phif):
    """Helper to do the wavelet transform using the fast wavelet domain transform.

    Args:
        data (numpy.ndarray): 1D numpy array data.
        Nf (int): Number of frequency bins.
        Nt (int): Number of time bins.
        phif (numpy.ndarray): 1D numpy array wavelet.

    Returns:
    -------
    2D numpy array
        Data in wavelet domain.
    """
    wave = np.zeros((Nt,Nf)) # wavelet wavepacket transform of the signal

    for m in range(0,Nf+1):
        DX = np.zeros(Nt,dtype=np.complex128)
        DX_assign_loop(m,Nt,Nf,DX,data,phif)
        DX_trans = np.fft.ifft(DX,Nt)
        DX_unpack_loop(m,Nt,Nf,DX_trans,wave)
    return wave

def transform_wavelet_freq_quadrature_helper(data,Nf,Nt,phif):
    """Helper to do the wavelet transform using the fast wavelet domain quadrature transform.

    Args:
        data (numpy.ndarray): 1D numpy array data.
        Nf (int): Number of frequency bins.
        Nt (int): Number of time bins.
        phif (numpy.ndarray): 1D numpy array wavelet.

    Returns:
        numpy.ndarray: 2D numpy array data in wavelet domain.
    """
    wave = np.zeros((Nt,Nf)) # wavelet wavepacket transform of the signal

    DX = np.zeros(Nt,dtype=np.complex128)
    for m in range(0,Nf+1):
        DX_assign_loop(m,Nt,Nf,DX,data,phif)
        DX_trans = np.fft.ifft(DX,Nt)
        DX_unpack_loop_quadrature(m,Nt,Nf,DX_trans,wave)
    return wave

@njit
def DX_assign_loop(m,Nt,Nf,DX,data,phif):
    """Helper for assigning DX in the main loop.

    Args:
        m (int): Frequency index.
        Nt (int): Number of time bins.
        Nf (int): Number of frequency bins.
        DX (numpy.ndarray): 1D complex numpy array DX.
        data (numpy.ndarray): 1D complex numpy array input data.
        phif (numpy.ndarray): 1D numpy array wavelet.
    """
    i_base = Nt//2
    jj_base = m*Nt//2

    if m==0 or m==Nf:
        #NOTE this term appears to be needed to recover correct constant (at least for m=0), but was previously missing
        DX[Nt//2] = phif[0]*data[m*Nt//2]/2.
        DX[Nt//2] = phif[0]*data[m*Nt//2]/2.
    else:
        DX[Nt//2] = phif[0]*data[m*Nt//2]
        DX[Nt//2] = phif[0]*data[m*Nt//2]

    for jj in range(jj_base+1-Nt//2,jj_base+Nt//2):
        j = np.abs(jj-jj_base)
        i = i_base-jj_base+jj
        if m==Nf and jj>jj_base:
            DX[i] = 0.
        elif m==0 and jj<jj_base:
            DX[i] = 0.
        elif j==0:
            continue
        else:
            DX[i] = phif[j]*data[jj]

@njit
def DX_unpack_loop(m,Nt,Nf,DX_trans,wave):
    """Helper for unpacking fftd DX in main loop.

    Args:
        m (int): Frequency index.
        Nt (int): Number of time bins.
        Nf (int): Number of frequency bins.
        DX_trans (numpy.ndarray): 1D complex numpy array of transformed DX.
        wave (numpy.ndarray): 2D numpy array for data in wavelet domain.
    """
    if m==0:
        #half of lowest and highest frequency bin pixels are redundant, so store them in even and odd components of m=0 respectively
        for n in range(0,Nt,2):
            wave[n,0] = np.real(DX_trans[n]*np.sqrt(2))
    elif m==Nf:
        for n in range(0,Nt,2):
            wave[n+1,0] = np.real(DX_trans[n]*np.sqrt(2))
    else:
        for n in range(0,Nt):
            if m%2:
                if (n+m)%2:
                    wave[n,m] = -np.imag(DX_trans[n])
                else:
                    wave[n,m] = np.real(DX_trans[n])
            else:
                if (n+m)%2:
                    wave[n,m] = np.imag(DX_trans[n])
                else:
                    wave[n,m] = np.real(DX_trans[n])

@njit
def DX_unpack_loop_quadrature(m,Nt,Nf,DX_trans,wave):
    """Helper for unpacking fftd DX in main loop.

    Args:
        m (int): Frequency index.
        Nt (int): Number of time bins.
        Nf (int): Number of frequency bins.
        DX_trans (numpy.ndarray): 1D complex numpy array of transformed DX.
        wave (numpy.ndarray): 2D numpy array for data in wavelet domain.
    """
    if m==0:
        #half of lowest and highest frequency bin pixels are redundant, so store them in even and odd components of m=0 respectively
        for n in range(0,Nt,2):
            wave[n,0] = np.real(DX_trans[n+1]*np.sqrt(2))
    elif m==Nf:
        for n in range(0,Nt,2):
            wave[n+1,0] = np.real(DX_trans[n+1]*np.sqrt(2))
    else:
        for n in range(0,Nt):
            if m%2:
                if (n+m)%2:
                    wave[n,m] = np.real(DX_trans[n])
                else:
                    wave[n,m] = -np.imag(DX_trans[n])
            else:
                if (n+m)%2:
                    wave[n,m] = np.real(DX_trans[n])
                else:
                    wave[n,m] = np.imag(DX_trans[n])
