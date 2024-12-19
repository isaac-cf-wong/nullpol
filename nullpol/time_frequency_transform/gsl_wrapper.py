import numpy as np
from numba import njit
from cffi import FFI

ffi = FFI()
ffi.cdef('double gsl_sf_beta_inc(double a, double b, double x);')
gsl = ffi.dlopen('libgsl.so')
gsl_gsl_sf_beta_inc = gsl.gsl_sf_beta_inc
 
@njit
def gsl_sf_beta_inc(a, b, x):
    return gsl_gsl_sf_beta_inc(a, b, x)