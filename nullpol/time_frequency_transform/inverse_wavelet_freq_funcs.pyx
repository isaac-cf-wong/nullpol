import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef unpack_wave_inverse(int m,
                          int Nt,
                          int Nf,
                          np.ndarray[np.float64_t,ndim=1] phif,
                          np.ndarray[np.complex128_t,ndim=1] fft_prefactor2s,
                          np.ndarray[np.complex128_t,ndim=1] res):
    """Helper for unpacking results of frequency domain inverse transform.

    Args:
        m (int): Frequency index.
        Nt (int): Number of time bins.
        Nf (int): Number of frequency bins.
        phif (1D numpy array): Wavelet.
        fft_prefactor2s (1D numpy array): Prefactors of FFT.
        res (1D numpy array): Result.
    """
    cdef int i_ind, i, ind3
    cdef int ind31, ind32, i1, i2
    if m==0 or m==Nf:
        for i_ind in range(0,Nt//2):
            i = np.abs(m*Nt//2-i_ind)#i_off+i_min2
            ind3 = (2*i)%Nt
            res[i] += fft_prefactor2s[ind3]*phif[i_ind]
        if m==Nf:
            i_ind = Nt//2
            i = np.abs(m*Nt//2-i_ind)#i_off+i_min2
            ind3 = 0
            res[i] += fft_prefactor2s[ind3]*phif[i_ind]
    else:
        ind31 = (Nt//2*m)%Nt
        ind32 = (Nt//2*m)%Nt
        for i_ind in range(0,Nt//2):
            i1 = Nt//2*m-i_ind
            i2 = Nt//2*m+i_ind
            #assert ind31 == i1%Nt
            #assert ind32 == i2%Nt
            res[i1] += fft_prefactor2s[ind31]*phif[i_ind]
            res[i2] += fft_prefactor2s[ind32]*phif[i_ind]
            ind31 -= 1
            ind32 += 1
            if ind31<0:
                ind31 = Nt-1
            if ind32==Nt:
                ind32 = 0

        res[Nt//2*m] = fft_prefactor2s[(Nt//2*m)%Nt]*phif[0]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.complex128_t,ndim=1] inverse_wavelet_freq_helper_fast(np.ndarray[np.float64_t,ndim=2] wave_in,
                                                                          np.ndarray[np.float64_t,ndim=1] phif,
                                                                          int Nf,
                                                                          int Nt):
    """Loop for inverse_wavelet_freq.

    Args:
        wave_in (2D numpy array): Input data in wavelet domain.
        phif (1D numpy array): Wavelet.
        Nf (int): Number of frequency bins.
        Nt (int ): Number of time bins.

    Returns:
        1D complex numpy array: Result.
    """
    cdef int ND=Nf*Nt

    cdef np.ndarray[np.complex128_t,ndim=1] prefactor2s = np.zeros(Nt,np.complex128)
    cdef np.ndarray[np.complex128_t,ndim=1] res = np.zeros(ND//2+1,dtype=np.complex128)
    cdef int m
    cdef np.ndarray[np.complex128_t,ndim=1] fft_prefactor2s
    for m in range(0,Nf+1):
        pack_wave_inverse(m,Nt,Nf,prefactor2s,wave_in)
        #with numba.objmode(fft_prefactor2s="complex128[:]"):
        fft_prefactor2s = np.fft.fft(prefactor2s)
        unpack_wave_inverse(m,Nt,Nf,phif,fft_prefactor2s,res)

    return res

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef pack_wave_inverse(int m,
                        int Nt,
                        int Nf,
                        np.ndarray[np.complex128_t,ndim=1] prefactor2s,
                        np.ndarray[np.float64_t,ndim=2] wave_in):
    """Helper for fast frequency domain inverse transform to preare for Fourier transform.

    Args:
        m (int): Frequency index.
        Nt (int): Number of time bins.
        Nf (int): Number of frequency bins.
        prefactor2s (1D complex numpy array): Prefactors for the 1D numpy array.
        wave_in (2D numpy array): Input data in wavelet domain.
    """
    cdef int n
    cdef complex mult2
    if m==0:
        for n in range(0,Nt):
            prefactor2s[n] = 1/np.sqrt(2)*wave_in[(2*n)%Nt,0]
    elif m==Nf:
        for n in range(0,Nt):
            prefactor2s[n] = 1/np.sqrt(2)*wave_in[(2*n)%Nt+1,0]
    else:
        for n in range(0,Nt):
            val = wave_in[n,m]
            if (n+m)%2:
                mult2 = -1j
            else:
                mult2 = 1

            prefactor2s[n] = mult2*val    