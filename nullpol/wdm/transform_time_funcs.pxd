cimport numpy as np

cpdef void assign_wdata(int i,
                        int K,
                        int ND,
                        int Nf,
                        np.ndarray[np.float64_t,ndim=1] wdata,
                        np.ndarray[np.float64_t,ndim=1] data_pad,
                        np.ndarray[np.float64_t,ndim=1] phi)

cpdef void pack_wave(int i,
                     int mult,
                     int Nf,
                     np.ndarray[np.float64_t,ndim=1] wdata_trans,
                     np.ndarray[np.float64_t,ndim=1] wave)

cpdef np.ndarray[np.float64_t,ndim=1] transform_wavelet_time_helper(np.ndarray[np.float64_t,ndim=1]data,
                                                                    int Nf,
                                                                    int Nt,
                                                                    np.ndarray[np.float64_t,ndim=1] phi,
                                                                    int mult)

cpdef np.ndarray[np.float64_t,ndim=1] phi_vec(int Nf,
                                              double nx=4.,
                                              int mult=16)                                                                                                                                                               