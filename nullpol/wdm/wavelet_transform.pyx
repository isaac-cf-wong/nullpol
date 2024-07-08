import numpy as np
cimport numpy as np
cimport cython
from nullpol.wdm.transform_freq_funcs cimport phitilde_vec_norm,transform_wavelet_freq_helper
from nullpol.wdm.transform_time_funcs cimport phi_vec,transform_wavelet_time_helper
from nullpol.wdm.inverse_wavelet_freq_funcs cimport inverse_wavelet_freq_helper_fast
from nullpol.wdm.inverse_wavelet_time_funcs cimport inverse_wavelet_time_helper_fast

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t,ndim=1] inverse_wavelet_time(np.ndarray[np.float64_t,ndim=2] wave_in,
                                                           int Nf,
                                                           int Nt,
                                                           double nx,
                                                           int mult):
    """fast inverse wavelet transform to time domain"""
    cdef int mult_ = np.min((mult,Nt//2)) #make sure K isn't bigger than ND
    cdef np.ndarray[np.float64_t,ndim=1] phi = phi_vec(Nf,nx=nx,mult=mult_)/2

    return inverse_wavelet_time_helper_fast(wave_in,phi,Nf,Nt,mult_)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t,ndim=1] inverse_wavelet_freq_time(np.ndarray[np.float64_t,ndim=2] wave_in,
                                                                int Nf,
                                                                int Nt,
                                                                double nx):
    """inverse wavlet transform to time domain via fourier transform of frequency domain"""
    cdef np.ndarray[np.complex128_t,ndim=1] res_f = inverse_wavelet_freq(wave_in,Nf,Nt,nx)
    return np.fft.irfft(res_f)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.complex128_t,ndim=1] inverse_wavelet_freq(np.ndarray[np.float64_t,ndim=2] wave_in,
                                                            int Nf,
                                                            int Nt,
                                                            double nx):
    """inverse wavelet transform to freq domain signal"""
    cdef np.ndarray[np.float64_t,ndim=1] phif = phitilde_vec_norm(Nf,Nt,nx)
    return inverse_wavelet_freq_helper_fast(wave_in,phif,Nf,Nt)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t,ndim=2] transform_wavelet_time(np.ndarray[np.float64_t,ndim=1] data,
                                                             int Nf,
                                                             int Nt,
                                                             double nx,
                                                             int mult):
    """do the wavelet transform in the time domain,
    note there can be significant leakage if mult is too small and the
    transform is only approximately exact if mult=Nt/2"""
    cdef int mult_ = np.min((mult,Nt//2)) #make sure K isn't bigger than ND
    cdef np.ndarray[np.float64_t,ndim=1] phi = phi_vec(Nf,nx,mult_)
    cdef np.ndarray[np.float64_t,ndim=2] wave = transform_wavelet_time_helper(data,Nf,Nt,phi,mult_)

    return wave

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t,ndim=2] transform_wavelet_freq_time(np.ndarray[np.float64_t,ndim=1] data,
                                                                  int Nf,
                                                                  int Nt,
                                                                  double nx):
    """transform time domain data into wavelet domain via fft and then frequency transform"""
    cdef np.ndarray[np.complex128_t,ndim=1] data_fft = np.fft.rfft(data)

    return transform_wavelet_freq(data_fft,Nf,Nt,nx)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t,ndim=2] transform_wavelet_freq(np.ndarray[np.complex128_t,ndim=1] data,
                                                             int Nf,
                                                             int Nt,
                                                             double nx):
    """do the wavelet transform using the fast wavelet domain transform"""
    cdef np.ndarray[np.float64_t,ndim=1] phif = 2/Nf*phitilde_vec_norm(Nf,Nt,nx)
    return transform_wavelet_freq_helper(data,Nf,Nt,phif)
