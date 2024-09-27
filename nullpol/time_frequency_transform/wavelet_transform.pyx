import numpy as np
cimport numpy as np
cimport cython
from .transform_freq_funcs cimport phitilde_vec_norm,transform_wavelet_freq_helper,transform_wavelet_freq_quadrature_helper
from .transform_time_funcs cimport phi_vec,transform_wavelet_time_helper
from .inverse_wavelet_freq_funcs cimport inverse_wavelet_freq_helper_fast
from .inverse_wavelet_time_funcs cimport inverse_wavelet_time_helper_fast

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t,ndim=1] inverse_wavelet_time(np.ndarray[np.float64_t,ndim=2] wave_in,
                                                           int Nf,
                                                           int Nt,
                                                           double nx,
                                                           int mult):
    """Fast inverse wavelet transform to time domain.

    Args:
        wave_in (2D numpy array): Data in wavelet domain.
        Nf (int): Number of frequency bins.
        Nt (int): Number of time bins.
        nx (float, optional): Steepness of the filter. Defaults to 4..
        mult (int, optional): mult. Defaults to 32.

    Returns:
        1D numpy array: Data in time domain.
    """
    cdef int mult_ = np.min((mult,Nt//2)) #make sure K isn't bigger than ND
    cdef np.ndarray[np.float64_t,ndim=1] phi = phi_vec(Nf,nx=nx,mult=mult_)/2

    cdef np.ndarray[np.float64_t,ndim=1] output = inverse_wavelet_time_helper_fast(wave_in,phi,Nf,Nt,mult_)
    cdef double scaling = 1. / np.sqrt(output.shape[0])
    return output * scaling

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t,ndim=1] inverse_wavelet_freq_time(np.ndarray[np.float64_t,ndim=2] wave_in,
                                                                int Nf,
                                                                int Nt,
                                                                double nx):
    """Inverse wavelet transform to time domain via Fourier transform of frequency domain.

    Args:
        wave_in (2D numpy array): Data in wavelet domain.
        Nf (int): Number of frequency bins.
        Nt (int): Number of time bins.
        nx (float, optional): Steepness of filter. Defaults to 4..

    Returns:
        1D numpy array: Data in time domain.
    """
    cdef np.ndarray[np.complex128_t,ndim=1] res_f = inverse_wavelet_freq(wave_in,Nf,Nt,nx)
    return np.fft.irfft(res_f)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.complex128_t,ndim=1] inverse_wavelet_freq(np.ndarray[np.float64_t,ndim=2] wave_in,
                                                            int Nf,
                                                            int Nt,
                                                            double nx):
    """Inverse wavelet transform to frequency domain signal.

    Args:
        wave_in (2D numpy array): Data in wavelet domain.
        Nf (int): Number of frequency bins.
        Nt (int): Number of time bins.
        nx (float, optional): Steepness of filter. Defaults to 4..

    Returns:
        1D numpy array: Data in time domain.
    """
    cdef np.ndarray[np.float64_t,ndim=1] phif = phitilde_vec_norm(Nf,Nt,nx)
    cdef np.ndarray[np.complex128_t,ndim=1] output = inverse_wavelet_freq_helper_fast(wave_in,phif,Nf,Nt)
    # Assume the length of the timeseries is even.
    cdef double scaling = 1. / np.sqrt((output.shape[0] - 1) * 2)
    return output * scaling

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t,ndim=2] transform_wavelet_time(np.ndarray[np.float64_t,ndim=1] data,
                                                             int Nf,
                                                             int Nt,
                                                             double nx,
                                                             int mult):
    """Do the wavelet transform in the time domain,
    note there can be significant leakage if mult is too small and the
    transform is only approximately exact if mult=Nt/2.

    Args:
        data (1D numpy array): Data in time domain.
        Nf (int): Number of frequency bins.
        Nt (int): Number of time bins.
        nx (float, optional): Steepness of filter. Defaults to 4..
        mult (int, optional): mult. Defaults to 32.

    Returns:
        2D numpy array: Data in wavelet domain.
    """
    cdef int mult_ = np.min((mult,Nt//2)) #make sure K isn't bigger than ND
    cdef np.ndarray[np.float64_t,ndim=1] phi = phi_vec(Nf,nx,mult_)
    cdef double scaling = np.sqrt(data.shape[0])
    cdef np.ndarray[np.float64_t,ndim=2] wave = transform_wavelet_time_helper(data,Nf,Nt,phi,mult_) * scaling

    return wave

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t,ndim=2] transform_wavelet_freq_time(np.ndarray[np.float64_t,ndim=1] data,
                                                                  int Nf,
                                                                  int Nt,
                                                                  double nx):
    """Transform time domain data into wavelet domain via FFT and then frequency transform.

    Args:
        data (1D numpy array): Data in time domain.
        Nf (int): Number of frequency bins.
        Nt (int): Number of time bins.
        nx (float, optional): Steepness of filter. Defaults to 4..

    Returns:
        2D numpy array: Data in wavelet domain.
    """
    cdef np.ndarray[np.complex128_t,ndim=1] data_fft = np.fft.rfft(data)
    return transform_wavelet_freq(data_fft,Nf,Nt,nx)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t,ndim=2] transform_wavelet_freq(np.ndarray[np.complex128_t,ndim=1] data,
                                                             int Nf,
                                                             int Nt,
                                                             double nx):
    """Do the wavelet transform using the fast wavelet domain transform.

    Args:
        data (1D complex numpy array): Data in frequency domain.
        Nf (int): Number of frequency bins.
        Nt (int): Number of time bins.
        nx (float, optional): Steepness of filter. Defaults to 4..

    Returns:
        2D numpy array: Data in wavelet domain.
    """
    cdef np.ndarray[np.float64_t,ndim=1] phif = 2/Nf*phitilde_vec_norm(Nf,Nt,nx)
    # Assume the length of the timeseries is even,
    cdef double scaling = np.sqrt((data.shape[0] - 1) * 2)
    return transform_wavelet_freq_helper(data,Nf,Nt,phif) * scaling

#@cython.boundscheck(False)
#@cython.wraparound(False)
#cpdef np.ndarray[np.float64_t,ndim=2] transform_wavelet_time_quadrature(np.ndarray[np.float64_t,ndim=1] data,
#                                                                        int Nf,
#                                                                        int Nt,
#                                                                        double nx,
#                                                                        int mult):
#    """do the wavelet transform in the time domain,
#    note there can be significant leakage if mult is too small and the
#    transform is only approximately exact if mult=Nt/2"""
#    cdef int mult_ = np.min((mult,Nt//2)) #make sure K isn't bigger than ND
#    cdef np.ndarray[np.float64_t,ndim=1] phi = phi_vec(Nf,nx,mult_)
#    cdef np.ndarray[np.float64_t,ndim=2] wave = transform_wavelet_time_quadrature_helper(data,Nf,Nt,phi,mult_)
#
#    return wave

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t,ndim=2] transform_wavelet_freq_time_quadrature(np.ndarray[np.float64_t,ndim=1] data,
                                                                             int Nf,
                                                                             int Nt,
                                                                             double nx):
    """Transform time domain data into wavelet domain via FFT and then frequency transform.

    Args:
        data (1D numpy array): Data in time domain.
        Nf (int): Number of frequency bins.
        Nt (int): Number of time bins.
        nx (float, optional): Steepness of filter. Defaults to 4..

    Returns:
        2D numpy array: Data in wavelet domain.
    """
    cdef np.ndarray[np.complex128_t,ndim=1] data_fft = np.fft.rfft(data)
    return transform_wavelet_freq_quadrature(data_fft,Nf,Nt,nx)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t,ndim=2] transform_wavelet_freq_quadrature(np.ndarray[np.complex128_t,ndim=1] data,
                                                                        int Nf,
                                                                        int Nt,
                                                                        double nx):
    """Do the wavelet transform using the fast wavelet domain quadrature transform.

    Args:
        data (1D complex numpy array): Data in frequency domain.
        Nf (int): Number of frequency bins.
        Nt (int): Number of time bins.
        nx (float, optional): Steepness of filter. Defaults to 4..

    Returns:
        2D numpy array: Data in wavelet domain.
    """
    cdef np.ndarray[np.float64_t,ndim=1] phif = 2/Nf*phitilde_vec_norm(Nf,Nt,nx)
    # Assume the length of the timeseries is even,
    cdef double scaling = np.sqrt((data.shape[0] - 1) * 2)
    return transform_wavelet_freq_quadrature_helper(data,Nf,Nt,phif) * scaling