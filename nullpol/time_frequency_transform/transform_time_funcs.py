"""helper functions for transform_time.py"""
import numpy as np
from numba import njit, prange
from .transform_freq_funcs import phitilde_vec

def transform_wavelet_time_helper(data,Nf,Nt,phi,mult):
    """Helper function to do the wavelet transform in the time domain.

    Parameters
    ----------
    data: 1D numpy array
        Data.
    Nf: int
        Number of frequency bins.
    Nt: int
        Number of time bins.
    phi: 1D numpy array
        Wavelet.
    mult: int
        mult

    Returns
    -------
    2D numpy array
        Data in wavelet domain.
    """    
    # the time domain data stream
    ND = Nf*Nt

    #mult, can cause bad leakage if it is too small but may be possible to mitigate
    # Filter is mult times pixel with in time

    K = mult*2*Nf

    # windowed data packets
    wdata = np.zeros(K)

    wave = np.zeros((Nt,Nf))  # wavelet wavepacket transform of the signal
    data_pad = np.zeros(ND+K)
    data_pad[:ND] = data
    data_pad[ND:ND+K] = data[:K]

    for i in range(0,Nt):
        assign_wdata(i,K,ND,Nf,wdata,data_pad,phi)
        wdata_trans = np.fft.rfft(wdata,K)
        pack_wave(i,mult,Nf,wdata_trans,wave)

    return wave

@njit
def assign_wdata(i,K,ND,Nf,wdata,data_pad,phi):
    """Assign wdata to be fftd in loop, data_pad needs K extra values on the right to loop.

    Parameters
    ----------
    i: int
        Time index.
    K: int
        Frequency cutoff.
    ND: int
        ND.
    Nf: int
        Number of frequency bins.
    wdata: 1D numpy array
        wdata.
    data_pad: 1D numpy array
        Padded data.
    phi: 1D numpy array
        Wavelet.
    """    
    #half_K = np.int64(K/2)
    jj = i*Nf-K//2
    if jj<0:
        jj += ND  # periodically wrap the data
    if jj>=ND:
        jj -= ND # periodically wrap the data
    for j in range(0,K):
        #jj = i*Nf-half_K+j
        wdata[j] = data_pad[jj]*phi[j]  # apply the window
        jj += 1
        #if jj==ND:
        #    jj -= ND # periodically wrap the data

@njit
def pack_wave(i,mult,Nf,wdata_trans,wave):
    """Pack fftd wdata into wave array.

    Parameters
    ----------
    i: int
        Time index.
    mult: int
        mult.
    Nf: int
        Number of frequency bins.
    wdata_trans: 1D complex numpy array
        wdata_trans.
    wave: 2D numpy array
        wdata.
    """    
    if i%2==0 and i<wave.shape[0]-1:
        #m=0 value at even Nt and
        wave[i,0] = np.real(wdata_trans[0])/np.sqrt(2)
        wave[i+1,0] = np.real(wdata_trans[Nf*mult])/np.sqrt(2)

    for j in range(1,Nf):
        if (i+j)%2:
            wave[i,j] = -np.imag(wdata_trans[j*mult])
        else:
            wave[i,j] = np.real(wdata_trans[j*mult])

def phi_vec(Nf,nx=4.,mult=16):
    """Get time domain phi as Fourier transform of phitilde_vec.

    Parameters
    ----------
    Nf: int
        Number of frequency bins.
    nx: float, optional
        Steepness of filter. Defaults to 4..
    mult: int, optional
        mult. Defaults to 16.

    Returns
    -------
    1D numpy array
        Time domain phi.
    """    
    #TODO fix mult

    OM = np.pi
    DOM = OM/Nf
    insDOM = 1./np.sqrt(DOM)
    K = mult*2*Nf
    half_K = mult*Nf#np.int64(K/2)

    dom = 2*np.pi/K  # max frequency is K/2*dom = pi/dt = OM

    DX = np.zeros(K,dtype=np.complex128)

    #zero frequency
    DX[0] =  insDOM

    DX = DX.copy()
    # postive frequencies
    DX[1:half_K+1] = phitilde_vec(dom*np.arange(1,half_K+1),Nf,nx)
    # negative frequencies
    DX[half_K+1:] = phitilde_vec(-dom*np.arange(half_K-1,0,-1),Nf,nx)
    DX = K*np.fft.ifft(DX,K)

    phi = np.zeros(K)
    phi[0:half_K] = np.real(DX[half_K:K])
    phi[half_K:] = np.real(DX[0:half_K])

    nrm = np.sqrt(K/dom)#*np.linalg.norm(phi)

    fac = np.sqrt(2.0)/nrm
    phi *= fac
    return phi