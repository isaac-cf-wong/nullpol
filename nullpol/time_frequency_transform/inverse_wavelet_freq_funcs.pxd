cimport numpy as np

cpdef unpack_wave_inverse(int m,
                          int Nt,
                          int Nf,
                          np.ndarray[np.float64_t,ndim=1] phif,
                          np.ndarray[np.complex128_t,ndim=1] fft_prefactor2s,
                          np.ndarray[np.complex128_t,ndim=1] res)

cpdef np.ndarray[np.complex128_t,ndim=1] inverse_wavelet_freq_helper_fast(np.ndarray[np.float64_t,ndim=2] wave_in,
                                                                          np.ndarray[np.float64_t,ndim=1] phif,
                                                                          int Nf,
                                                                          int Nt)

cpdef pack_wave_inverse(int m,
                        int Nt,
                        int Nf,
                        np.ndarray[np.complex128_t,ndim=1] prefactor2s,
                        np.ndarray[np.float64_t,ndim=2] wave_in)                                                                                                         