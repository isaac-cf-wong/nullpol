import numpy as np
cimport numpy as np
cimport cython
from .transform_freq_funcs import phitilde_vec


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void assign_wdata(int i,
                        int K,
                        int ND,
                        int Nf,
                        np.ndarray[np.float64_t,ndim=1] wdata,
                        np.ndarray[np.float64_t,ndim=1] data_pad,
                        np.ndarray[np.float64_t,ndim=1] phi):
    """Assign wdata to be fftd in loop, data_pad needs K extra values on the right to loop.

    Args:
        i (int): Time index.
        K (int): Frequency cutoff.
        ND (int): ND.
        Nf (int): Number of frequency bins.
        wdata (1D numpy array): wdata.
        data_pad (1D numpy array): Padded data.
        phi (1D numpy array): Wavelet.
    """
    #half_K = np.int64(K/2)
    cdef int jj = i*Nf-K//2
    cdef int j
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

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void pack_wave(int i,
                     int mult,
                     int Nf,
                     np.ndarray[np.complex128_t,ndim=1] wdata_trans,
                     np.ndarray[np.float64_t,ndim=2] wave):
    """Pack fftd wdata into wave array.

    Args:
        i (int): Time index.
        mult (int): mult.
        Nf (int): Number of frequency bins.
        wdata_trans (1D complex numpy array): wdata_trans.
        wave (2D numpy array): wdata.
    """
    if i%2==0 and i<wave.shape[0]-1:
        #m=0 value at even Nt and
        wave[i,0] = np.real(wdata_trans[0])/np.sqrt(2)
        wave[i+1,0] = np.real(wdata_trans[Nf*mult])/np.sqrt(2)
    cdef int j
    for j in range(1,Nf):
        if (i+j)%2:
            wave[i,j] = -np.imag(wdata_trans[j*mult])
        else:
            wave[i,j] = np.real(wdata_trans[j*mult])

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t,ndim=2] transform_wavelet_time_helper(np.ndarray[np.float64_t,ndim=1]data,
                                                                    int Nf,
                                                                    int Nt,
                                                                    np.ndarray[np.float64_t,ndim=1] phi,
                                                                    int mult):
    """Helper function to do the wavelet transform in the time domain.

    Args:
        data (1D numpy array): Data.
        Nf (int): Number of frequency bins.
        Nt (int): Number of time bins.
        phi (1D numpy array): Wavelet.
        mult (int): mult

    Returns:
        2D numpy array: Data in wavelet domain.
    """
    # the time domain data stream
    cdef int ND = Nf*Nt

    #mult, can cause bad leakage if it is too small but may be possible to mitigate
    # Filter is mult times pixel with in time

    cdef int K = mult*2*Nf

    # windowed data packets
    cdef np.ndarray[np.float64_t,ndim=1] wdata = np.zeros(K,dtype=np.float64)

    cdef np.ndarray[np.float_t,ndim=2] wave = np.zeros((Nt,Nf),dtype=np.float64)  # wavelet wavepacket transform of the signal
    cdef np.ndarray[np.float_t,ndim=1] data_pad = np.zeros(ND+K,dtype=np.float64)
    data_pad[:ND] = data
    data_pad[ND:ND+K] = data[:K]
    cdef int i

    for i in range(0,Nt):
        assign_wdata(i,K,ND,Nf,wdata,data_pad,phi)
        wdata_trans = np.fft.rfft(wdata,K)
        pack_wave(i,mult,Nf,wdata_trans,wave)

    return wave

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t,ndim=1] phi_vec(int Nf,
                                              double nx,
                                              int mult):
    """Get time domain phi as Fourier transform of phitilde_vec.

    Args:
        Nf (int): Number of frequency bins.
        nx (float, optional): Steepness of filter. Defaults to 4..
        mult (int, optional): mult. Defaults to 16.

    Returns:
        1D numpy array: Time domain phi.
    """
    #TODO fix mult

    cdef double OM = np.pi
    cdef double DOM = OM/Nf
    cdef double insDOM = 1./np.sqrt(DOM)
    cdef int K = mult*2*Nf
    cdef int half_K = mult*Nf#np.int64(K/2)

    cdef double dom = 2*np.pi/K  # max frequency is K/2*dom = pi/dt = OM

    cdef np.ndarray[np.complex128_t,ndim=1] DX = np.zeros(K,dtype=np.complex128)

    #zero frequency
    DX[0] =  insDOM

    # postive frequencies
    DX[1:half_K+1] = phitilde_vec(dom*np.arange(1,half_K+1),Nf,nx)
    # negative frequencies
    DX[half_K+1:] = phitilde_vec(-dom*np.arange(half_K-1,0,-1),Nf,nx)
    DX = np.fft.ifft(DX,K)*K

    cdef np.ndarray[np.float64_t,ndim=1] phi = np.zeros(K,dtype=np.float64)
    phi[0:half_K] = np.real(DX[half_K:K])
    phi[half_K:] = np.real(DX[0:half_K])

    cdef double nrm = np.sqrt(K/dom)#*np.linalg.norm(phi)

    cdef double fac = np.sqrt(2.0)/nrm
    phi *= fac
    return phi