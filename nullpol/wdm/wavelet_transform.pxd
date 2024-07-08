import numpy as np
cimport numpy as np

cpdef np.ndarray[np.float64_t,ndim=1] inverse_wavelet_time(np.ndarray[np.float64_t,ndim=2] wave_in,
                                                           int Nf,
                                                           int Nt,
                                                           double nx,
                                                           int mult)

cpdef np.ndarray[np.float64_t,ndim=1] inverse_wavelet_freq_time(np.ndarray[np.float64_t,ndim=2] wave_in,
                                                                int Nf,
                                                                int Nt,
                                                                double nx)

cpdef np.ndarray[np.complex128_t,ndim=1] inverse_wavelet_freq(np.ndarray[np.float64_t,ndim=2] wave_in,
                                                            int Nf,
                                                            int Nt,
                                                            double nx)

cpdef np.ndarray[np.float64_t,ndim=2] transform_wavelet_time(np.ndarray[np.float64_t,ndim=1] data,
                                                             int Nf,
                                                             int Nt,
                                                             double nx,
                                                             int mult)

cpdef np.ndarray[np.float64_t,ndim=2] transform_wavelet_freq_time(np.ndarray[np.float64_t,ndim=1] data,
                                                                  int Nf,
                                                                  int Nt,
                                                                  double nx)

cpdef np.ndarray[np.float64_t,ndim=2] transform_wavelet_freq(np.ndarray[np.complex128_t,ndim=1] data,
                                                             int Nf,
                                                             int Nt,
                                                             double nx)
