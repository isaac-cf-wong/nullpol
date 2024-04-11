import numpy as np
cimport numpy as np
cimport cython
from cython_gsl cimport gsl_sf_beta_inc
from libc.math cimport cos, pi, sqrt, log10


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t,ndim=1] compute_frequency_domain_wavelet(np.ndarray[np.float64_t,ndim=1] frequency_array,
                                                                       double bandwidth_of_top_flat_region,
                                                                       double total_bandwidth,
                                                                       double sharpness):
    cdef int i
    cdef np.ndarray[np.float64_t,ndim=1] output = np.zeros_like(frequency_array, dtype=np.float64)
    cdef double bandwidth_of_transition_region = total_bandwidth - bandwidth_of_top_flat_region
    cdef double domega = np.pi * total_bandwidth * 2
    cdef normalization_factor = 1 / np.sqrt(domega)
    # Construct the wavelet
    cdef double frequency_array_delta_f = frequency_array[1] - frequency_array[0]
    cdef int k_half_top_bandwidth = int(bandwidth_of_top_flat_region / frequency_array_delta_f / 2)
    cdef int k_bandwidth_of_transition_region = int(bandwidth_of_transition_region / frequency_array_delta_f)
    cdef int k_high = k_half_top_bandwidth + k_bandwidth_of_transition_region
    cdef double x
    cdef double beta_inc_norm = gsl_sf_beta_inc(sharpness,sharpness,1.)
    # The frequency components in the flat region
    for i in range(k_half_top_bandwidth):
        output[i] = normalization_factor
    for i in range(k_half_top_bandwidth, k_high):
        x = <double>(i - k_half_top_bandwidth)/k_bandwidth_of_transition_region
        output[i] = normalization_factor*cos(gsl_sf_beta_inc(sharpness,sharpness,x)/beta_inc_norm*pi/2)
    return output

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t,ndim=1] compute_time_domain_wavelet(np.ndarray[np.float64_t,ndim=1] time_array,
                                                                  double bandwidth_of_top_flat_region,
                                                                  double total_bandwidth,
                                                                  double sharpness):
    cdef int i
    cdef double delta_t = time_array[1] - time_array[0]
    cdef int tlen = time_array.shape[0]
    cdef np.ndarray[np.float64_t,ndim=1] frequency_array = np.fft.rfftfreq(tlen, delta_t)
    cdef np.ndarray[np.float64_t,ndim=1] frequency_domain_wavelet_array = compute_frequency_domain_wavelet(frequency_array, bandwidth_of_top_flat_region, total_bandwidth, sharpness)
    cdef np.ndarray[np.float64_t,ndim=1] time_domain_wavelet_array = np.fft.irfft(frequency_domain_wavelet_array)
    # Normalize the time domain wavelet
    # The sum of the wavelet squared is 1.
    cdef double norm = sqrt(2. * pi / delta_t)
    for i in range(tlen):
        time_domain_wavelet_array[i] *= norm
    return time_domain_wavelet_array

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int find_time_domain_wavelet_cutoff_time(np.ndarray[np.float64_t,ndim=1] time_domain_wavelet_array,
                                               double precision):
    cdef int i
    cdef int tlen = time_domain_wavelet_array.shape[0]
    cdef int tlen_over_2 = tlen//2
    # Compute the sum of energy squared
    cdef np.ndarray[np.float64_t,ndim=1] energy_2_arr = np.empty(tlen_over_2,dtype=np.float64)
    cdef double energy_2 = 0.
    for i in range(tlen_over_2):
        energy_2 += time_domain_wavelet_array[i]**2
        energy_2_arr[i] = energy_2
    # Find the index that the precision passes the threshold
    cdef int t_cut_idx = tlen_over_2-1
    for i in range(tlen_over_2-1):
        if -log10(1. - energy_2_arr[i]/energy_2) >= precision:
            t_cut_idx = i
            break
    return t_cut_idx

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_time_frequency_transform(np.ndarray time_domain_strain,
                                       double sampling_rate,
                                       double bandwidth_of_top_flat_region,
                                       double total_bandwidth,
                                       double sharpness,
                                       double precision=6):
    # Maximum frequency
    cdef double maximum_frequency = sampling_rate/2
    # Check whether the maximum frequency can be divided evenly by the total bandwidth
    if maximum_frequency%total_bandwidth != 0:
        raise ValueError("The total bandwidth must be a divisor of the maximum frequency.")
    # Length of the time domain strain
    cdef int tlen = time_domain_strain.shape[0]
    # Check whether the length of the time domain strain is even
    if tlen%2 != 0:
        raise ValueError("The length of the time domain strain must be even.")
    # A two-dimensional array to store the Fourier transform of the X vector
    # The first dimension is the time index
    # The second dimension is the frequency index
    # Number of frequency components
    cdef int kmax = int(maximum_frequency/total_bandwidth)
    cdef int nfreq = kmax+1
    # Number of time components
    cdef int ntime = int(tlen/kmax)
    cdef np.ndarray[np.float64_t,ndim=2] X = np.zeros((ntime,2*kmax), dtype=np.float64)
    cdef np.ndarray[np.complex128_t,ndim=2] Xf = np.empty((ntime, nfreq), dtype=np.complex128)
    cdef int X_kmax_plus_one = int(tlen/2/kmax)
    # Construct the time domain wavelet
    cdef np.ndarray[np.float64_t,ndim=1] time_array = np.arange(0,tlen) / sampling_rate
    cdef np.ndarray[np.float64_t,ndim=1] time_domain_wavelet = compute_time_domain_wavelet(time_array,
                                                                                           bandwidth_of_top_flat_region,
                                                                                           total_bandwidth,
                                                                                           sharpness)
    # Find the time cutoff
    cdef int wavelet_time_idx_cutoff = find_time_domain_wavelet_cutoff_time(time_domain_wavelet,
                                                                            precision)
    # Output array
    cdef np.ndarray[np.float64_t,ndim=2] output = np.zeros((ntime,nfreq),dtype=np.float64)
    cdef int i
    for i in range(ntime):
        # Compute the Xn vector
        pass
