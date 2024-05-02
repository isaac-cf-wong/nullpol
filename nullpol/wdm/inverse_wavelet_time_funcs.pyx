import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void unpack_time_wave_helper(int n,
                                   int Nf,
                                   int Nt,
                                   int K,
                                   np.ndarray[np.float64_t,ndim=1] phis,
                                   np.ndarray[np.float64_t,ndim=1] fft_fin_real,
                                   np.ndarray[np.float64_t,ndim=1] res):
    """helper for time domain wavelet transform to unpack wavelet domain coefficients"""
    cdef int ND = Nf*Nt

    cdef int idxf = (-K//2+n*Nf+ND)%(2*Nf)
    cdef int k = (-K//2+n*Nf)%ND
    cdef int k_ind
    cdef double res_loc
    for k_ind in range(0,K):
        res_loc = fft_fin_real[idxf]
        res[k] += phis[k_ind]*res_loc
        idxf+=1
        k += 1

        if idxf==2*Nf:
            idxf = 0
        if k==ND:
            k = 0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void unpack_time_wave_helper_compact(int n,
                                           int Nf,
                                           int Nt,
                                           int K,
                                           np.ndarray[np.float64_t,ndim=1] phis,
                                           np.ndarray[np.float64_t,ndim=1] fft_fin,
                                           np.ndarray[np.float64_t,ndim=1] res):
    """helper for time domain wavelet transform to unpack wavelet domain coefficients
    in compact representation where cosine and sine parts are real and imaginary parts"""
    cdef int ND = Nf*Nt
    cdef np.ndarray[np.float64_t,ndim=1] fft_fin_real = np.zeros(4*Nf,dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=1] fft_fin_imag = np.zeros(4*Nf,dtype=np.float64)
    cdef int itrf
    for itrf in range(0,2*Nf):
        fft_fin_real[itrf] = np.real(fft_fin[itrf])
        fft_fin_real[itrf+2*Nf] = fft_fin_real[itrf]
        fft_fin_imag[itrf] = np.imag(fft_fin[(itrf+Nf)%(2*Nf)])
        fft_fin_imag[itrf+2*Nf] = fft_fin_imag[itrf]

    cdef int idxf1_base = (-K//2+n*Nf+ND)%(2*Nf)
    cdef int k1_base = (-K//2+n*Nf)%ND
    cdef int k_ind, idxf1_add, idxf1, k_ind_loc, k1
    for k_ind in range(0,K,2*Nf):
        for idxf1_add in range(0,2*Nf):
            idxf1 = idxf1_base+idxf1_add#k_ind%(2*Nf)
            k_ind_loc = k_ind+idxf1_add
            k1 = k1_base+k_ind_loc

            res[k1] += phis[k_ind_loc]*fft_fin_real[idxf1]
            res[k1+Nf] += phis[k_ind_loc]*fft_fin_imag[idxf1]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void pack_wave_time_helper(int n,
                                 int Nf,
                                 int Nt,
                                 np.ndarray[np.float64_t,ndim=1] wave_in,
                                 np.ndarray[np.complex128_t,ndim=1] afins):
    """helper for time domain transform to pack wavelet domain coefficients"""
    if n%2==0:
        #assign highest and lowest bin correctly
        afins[0] = np.sqrt(2)*wave_in[n,0]
        if n+1<Nt:
            afins[Nf] = np.sqrt(2)*wave_in[n+1,0]
    else:
        afins[0] = 0.
        afins[Nf] = 0.

    cdef int idxm
    for idxm in range(0,Nf//2-1):
        if n%2:
            afins[2*idxm+2] = 1j*wave_in[n,2*idxm+2]
            afins[2*Nf-2*idxm-2] = -1j*wave_in[n,2*idxm+2]
        else:
            afins[2*idxm+2] = 1*wave_in[n,2*idxm+2]
            afins[2*Nf-2*idxm-2] = 1*wave_in[n,2*idxm+2]

    for idxm in range(0,Nf//2):
        if n%2:
            afins[2*idxm+1] = -1*wave_in[n,2*idxm+1]
            afins[2*Nf-2*idxm-1] = -1*wave_in[n,2*idxm+1]
        else:
            afins[2*idxm+1] = 1j*wave_in[n,2*idxm+1]
            afins[2*Nf-2*idxm-1] = -1j*wave_in[n,2*idxm+1]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void pack_wave_time_helper_compact(int n,
                                         int Nf,
                                         int Nt,
                                         np.ndarray[np.float64_t,ndim=1] wave_in,
                                         np.ndarray[np.complex128_t,ndim=1] afins):
    """helper for time domain transform to pack wavelet domain coefficients
    in packed representation with odd and even coefficients in real and imaginary pars"""
    afins[0] = np.sqrt(2)*wave_in[n,0]
    if n+1<Nt:
        afins[Nf] = np.sqrt(2)*wave_in[n+1,0]
    cdef int idxm
    for idxm in range(0,Nf-2,2):
        afins[idxm+2] = wave_in[n,idxm+2]-wave_in[n+1,idxm+2]
        afins[2*Nf-idxm-2] = wave_in[n,idxm+2]+wave_in[n+1,idxm+2]

        afins[idxm+1] = 1j*(wave_in[n,idxm+1]-wave_in[n+1,idxm+1])
        afins[2*Nf-idxm-1] = -1j*(wave_in[n,idxm+1]+wave_in[n+1,idxm+1])

    afins[Nf-1] = 1j*(wave_in[n,Nf-1]-wave_in[n+1,Nf-1])
    afins[Nf+1] = -1j*(wave_in[n,Nf-1]+wave_in[n+1,Nf-1])

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t,ndim=1] inverse_wavelet_time_helper_fast(np.ndarray[np.float64_t,ndim=1] wave_in,
                                                                       np.ndarray[np.float64_t,ndim=1] phi,
                                                                       int Nf,
                                                                       int Nt,
                                                                       int mult):
    """helper loop for fast inverse wavelet transform"""
    cdef int ND = Nf*Nt
    cdef int K = mult*2*Nf
    #res = np.zeros(ND)

    #extend this array, we can use wrapping boundary conditions at end
    cdef np.ndarray[np.float64_t,ndim=1] res = np.zeros(ND+K+Nf, dtype=np.float64)
    cdef np.ndarray[np.complex128_t,ndim=1] afins = np.zeros(2*Nf,dtype=np.complex128)
    cdef np.ndarray[np.complex128_t,ndim=1] ffts_fin
    cdef int n

    for n in range(0,Nt):
        #old unpacked way, should still work but is necessarily slower,
        #might be more comparable if it could be written as an irfft instead
        #pack_wave_time_helper(n,Nf,Nt,wave_in,afins)
        #ffts_fin_real = np.real(fft.fft(afins))
        #unpack_time_wave_helper(n,Nf,Nt,K,phi,ffts_fin_real,res)

        #we can pack both the sin and cos parts into the real and imaginary parts of the same transform so we only need to do every other one
        #this currently assumes Nt is even
        if n%2==0:
            pack_wave_time_helper_compact(n,Nf,Nt,wave_in,afins)
            ffts_fin = np.fft.fft(afins)
            unpack_time_wave_helper_compact(n,Nf,Nt,K,phi,ffts_fin,res)

    #wrap boundary conditions
    res[:min(K+Nf,ND)] += res[ND:min(ND+K+Nf,2*ND)]
    if K+Nf>ND:
        res[:K+Nf-ND] += res[2*ND:ND+K*Nf]

    res = res[:ND]

    return res
