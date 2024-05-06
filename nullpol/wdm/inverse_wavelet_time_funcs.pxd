cimport numpy as np

cpdef void unpack_time_wave_helper(int n,
                                   int Nf,
                                   int Nt,
                                   int K,
                                   np.ndarray[np.float64_t,ndim=1] phis,
                                   np.ndarray[np.float64_t,ndim=1] fft_fin_real,
                                   np.ndarray[np.float64_t,ndim=1] res)

cpdef void unpack_time_wave_helper_compact(int n,
                                           int Nf,
                                           int Nt,
                                           int K,
                                           np.ndarray[np.float64_t,ndim=1] phis,
                                           np.ndarray[np.complex128_t,ndim=1] fft_fin,
                                           np.ndarray[np.float64_t,ndim=1] res)

cpdef void pack_wave_time_helper(int n,
                                 int Nf,
                                 int Nt,
                                 np.ndarray[np.float64_t,ndim=2] wave_in,
                                 np.ndarray[np.complex128_t,ndim=1] afins)

cpdef void pack_wave_time_helper_compact(int n,
                                         int Nf,
                                         int Nt,
                                         np.ndarray[np.float64_t,ndim=2] wave_in,
                                         np.ndarray[np.complex128_t,ndim=1] afins)

cpdef np.ndarray[np.float64_t,ndim=1] inverse_wavelet_time_helper_fast(np.ndarray[np.float64_t,ndim=2] wave_in,
                                                                       np.ndarray[np.float64_t,ndim=1] phi,
                                                                       int Nf,
                                                                       int Nt,
                                                                       int mult)                                                                                                                                                                                                                                                                   