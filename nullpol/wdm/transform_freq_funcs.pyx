import numpy as np
cimport numpy as np
cimport cython
from cython_gsl cimport gsl_sf_beta_inc


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t,ndim=1] phitilde_vec(np.ndarray[np.float64_t,ndim=1] om,
                                                   int Nf,
                                                   double nx):
    """compute phitilde, om i array, nx is filter steepness, defaults to 4."""
    cdef double OM = np.pi  #Nyquist angular frequency
    cdef double DOM = OM/Nf #2 pi times DF
    cdef double insDOM = 1./np.sqrt(DOM)
    cdef double B = OM/(2*Nf)
    cdef double A = (DOM-B)/2
    cdef np.ndarray[np.float64_t,ndim=1] z = np.zeros(om.size,dtype=np.float64)

    cdef np.ndarray[np.uint8_t,ndim=1,cast=True] mask = (np.abs(om)>= A)&(np.abs(om)<A+B)

    cdef np.ndarray[np.float64_t,ndim=1] x = (np.abs(om[mask])-A)/B
    cdef np.ndarray[np.float64_t,ndim=1] y = np.zeros(x.shape[0],dtype=np.float64)
    cdef int i
    cdef double beta_inc_norm = gsl_sf_beta_inc(nx,nx,1.)
    for i in range(y.shape[0]):
        y[i] = gsl_sf_beta_inc(nx,nx,x[i])/beta_inc_norm
    z[mask] = insDOM*np.cos(np.pi/2.*y)

    z[np.abs(om)<A] = insDOM
    return z

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t,ndim=1] phitilde_vec_norm(int Nf,
                                                        int Nt,
                                                        double nx):
    """normalize phitilde as needed for inverse frequency domain transform"""
    cdef int ND = Nf*Nt
    cdef np.ndarray[np.float64_t,ndim=1] oms = np.arange(0,Nt//2+1,dtype=np.float64) * (2.*np.pi/ND)
    cdef np.ndarray[np.float64_t,ndim=1] phif = phitilde_vec(oms,Nf,nx)
    #nrm should be 1
    cdef double nrm = np.sqrt((2*np.sum(phif[1:]**2)+phif[0]**2)*2*np.pi/ND)
    nrm /= np.pi**(3/2)/np.pi
    phif /= nrm
    return phif

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void tukey(np.ndarray[np.float64_t,ndim=1] data,
                 double alpha,
                 int N):
    """apply tukey window function to data"""
    cdef int imin = np.int64(alpha*(N-1)/2)
    cdef int imax = np.int64((N-1)*(1-alpha/2))
    cdef int Nwin = N-imax
    cdef int i
    cdef double f_mult
    for i in range(0,N):
        f_mult = 1.0
        if i<imin:
            f_mult = 0.5*(1.+np.cos(np.pi*(i/imin-1.)))
        if i>imax:
            f_mult = 0.5*(1.+np.cos(np.pi/Nwin*(i-imax)))
        data[i] *= f_mult

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void DX_assign_loop(int m,
                          int Nt,
                          int Nf,
                          np.ndarray[np.complex128_t,ndim=1] DX,
                          np.ndarray[np.complex128_t,ndim=1] data,
                          np.ndarray[np.float64_t,ndim=1] phif):
    """helper for assigning DX in the main loop"""
    cdef int i_base = Nt//2
    cdef int jj_base = m*Nt//2

    if m==0 or m==Nf:
        #NOTE this term appears to be needed to recover correct constant (at least for m=0), but was previously missing
        DX[Nt//2] = phif[0]*data[m*Nt//2]/2.
        DX[Nt//2] = phif[0]*data[m*Nt//2]/2.
    else:
        DX[Nt//2] = phif[0]*data[m*Nt//2]
        DX[Nt//2] = phif[0]*data[m*Nt//2]
    cdef int jj, j, i
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

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void DX_unpack_loop(int m,
                          int Nt,
                          int Nf,
                          np.ndarray[np.complex128_t,ndim=1] DX_trans,
                          np.ndarray[np.float64_t,ndim=2] wave):
    """helper for unpacking fftd DX in main loop"""
    cdef int n
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

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void DX_unpack_loop_quadrature(int m,
                                     int Nt,
                                     int Nf,
                                     np.ndarray[np.complex128_t,ndim=1] DX_trans,
                                     np.ndarray[np.float64_t,ndim=2] wave):
    """helper for unpacking fftd DX in main loop"""
    cdef int n
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

cpdef np.ndarray[np.float64_t,ndim=2] transform_wavelet_freq_helper(np.ndarray[np.complex128_t,ndim=1] data,
                                                                    int Nf,
                                                                    int Nt,
                                                                    np.ndarray[np.float64_t,ndim=1] phif):
    """helper to do the wavelet transform using the fast wavelet domain transform"""
    cdef np.ndarray[np.float64_t,ndim=2] wave = np.zeros((Nt,Nf),dtype=np.float64) # wavelet wavepacket transform of the signal

    cdef np.ndarray[np.complex128_t,ndim=1] DX = np.zeros(Nt,dtype=np.complex128)
    cdef int m
    cdef np.ndarray[np.complex128_t,ndim=1] DX_trans
    for m in range(0,Nf+1):
        DX_assign_loop(m,Nt,Nf,DX,data,phif)
        DX_trans = np.fft.ifft(DX,Nt)
        DX_unpack_loop(m,Nt,Nf,DX_trans,wave)
    return wave

cpdef np.ndarray[np.float64_t,ndim=2] transform_wavelet_freq_quadrature_helper(np.ndarray[np.complex128_t,ndim=1] data,
                                                                               int Nf,
                                                                               int Nt,
                                                                               np.ndarray[np.float64_t,ndim=1] phif):
    """helper to do the wavelet transform using the fast wavelet domain transform"""
    #cdef np.ndarray[np.float64_t,ndim=2] wave = np.zeros((Nt,Nf),dtype=np.float64) # wavelet wavepacket transform of the signal
    cdef np.ndarray[np.float64_t,ndim=2] wave_q = np.zeros((Nt,Nf),dtype=np.float64) # quadrature wavelet wavepacket transform of the signal

    cdef np.ndarray[np.complex128_t,ndim=1] DX = np.zeros(Nt,dtype=np.complex128)
    cdef int m
    cdef np.ndarray[np.complex128_t,ndim=1] DX_trans
    for m in range(0,Nf+1):
        DX_assign_loop(m,Nt,Nf,DX,data,phif)
        DX_trans = np.fft.ifft(DX,Nt)
        #DX_unpack_loop(m,Nt,Nf,DX_trans,wave)
        DX_unpack_loop_quadrature(m,Nt,Nf,DX_trans,wave_q)
    return wave_q
