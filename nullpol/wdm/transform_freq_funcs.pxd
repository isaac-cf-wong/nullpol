cimport numpy as np

cpdef np.ndarray[np.float64_t,ndim=1] phitilde_vec(np.ndarray[np.float64_t,ndim=1] om,
                                                   int Nf,
                                                   double nx)

cpdef np.ndarray[np.float64_t,ndim=1] phitilde_vec_norm(int Nf,
                                                        int Nt,
                                                        double nx)

cpdef void tukey(np.ndarray[np.float64_t,ndim=1] data,
                 double alpha,
                 int N)

cpdef void DX_assign_loop(int m,
                          int Nt,
                          int Nf,
                          np.ndarray[np.complex128_t,ndim=1] DX,
                          np.ndarray[np.complex128_t,ndim=1] data,
                          np.ndarray[np.float64_t,ndim=1] phif)

cpdef void DX_unpack_loop(int m,
                          int Nt,
                          int Nf,
                          np.ndarray[np.complex128_t,ndim=1] DX_trans,
                          np.ndarray[np.float64_t,ndim=2] wave)

cpdef void DX_unpack_loop_quadrature(int m,
                                     int Nt,
                                     int Nf,
                                     np.ndarray[np.complex128_t,ndim=1] DX_trans,
                                     np.ndarray[np.float64_t,ndim=2] wave)

cpdef np.ndarray[np.float64_t,ndim=2] transform_wavelet_freq_helper(np.ndarray[np.complex128_t,ndim=1] data,
                                                                    int Nf,
                                                                    int Nt,
                                                                    np.ndarray[np.float64_t,ndim=1] phif)

cpdef np.ndarray[np.float64_t,ndim=2] transform_wavelet_freq_quadrature_helper(np.ndarray[np.complex128_t,ndim=1] data,
                                                                               int Nf,
                                                                               int Nt,
                                                                               np.ndarray[np.float64_t,ndim=1] phif)
