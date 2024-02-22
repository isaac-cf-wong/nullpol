###################################################################
# Implementation of the Wilson-Dauechies time-frequency transform
#
# Reference:
# https://iopscience.iop.org/article/10.1088/1742-6596/363/1/012032/pdf
###################################################################
import numpy as np
from scipy.special import betainc
# Plotting for inspection
import matplotlib.pyplot as plt

def frequency_domain_wavelet(frequency_array, bandwidth_of_top_flat_region, total_bandwidth, sharpness):
    output = np.zeros_like(frequency_array)
    bandwidth_of_transition_region = total_bandwidth - bandwidth_of_top_flat_region
    domega = 2 * np.pi * total_bandwidth
    normalization_factor = 1 / np.sqrt(domega)
    
    # Construct the wavelet
    frequency_array_delta_f = frequency_array[1] - frequency_array[0]
    k_half_top_bandwidth = int(bandwidth_of_top_flat_region / frequency_array_delta_f / 2)
    k_bandwidth_of_transition_region = int(bandwidth_of_transition_region / frequency_array_delta_f)
    k_high = k_half_top_bandwidth + k_bandwidth_of_transition_region
    # The frequency components in the flat region
    output[:k_half_top_bandwidth] = normalization_factor
    # The frequency components in the transition region
    ## The k indices of the components
    k_transition = np.arange(k_half_top_bandwidth, k_high)
    output[k_transition] = normalization_factor*np.cos(betainc(sharpness, sharpness, (k_transition - k_half_top_bandwidth) / k_bandwidth_of_transition_region)*np.pi/2)
    return output

def time_domain_wavelet(time_array, bandwidth_of_top_flat_region, total_bandwidth, sharpness):
    frequency_array = np.fft.rfftfreq(len(time_array), time_array[1] - time_array[0])
    frequency_domain_wavelet_array = frequency_domain_wavelet(frequency_array, bandwidth_of_top_flat_region, total_bandwidth, sharpness)
    time_domain_wavelet_array = np.fft.irfft(frequency_domain_wavelet_array)
    return time_domain_wavelet_array

def time_frequency_transform(time_domain_strain,
                             sampling_rate,
                             bandwidth_of_top_flat_region,
                             total_bandwidth,
                             sharpness):
    # Maximum frequency
    maximum_frequency = sampling_rate / 2
    # Check whether the maximum frequency can be divided evenly by the total bandwidth
    if maximum_frequency % total_bandwidth != 0:
        raise ValueError(f"The total bandwidth = {total_bandwidth} must be a divisor of the maximum frequency = {maximum_frequency}")
    # Length of the time domain strain
    tlen = len(time_domain_strain)    
    # Check whether the length of the time domain strain is even
    if tlen % 2 != 0:
        raise ValueError(f"The length of the time domain strain = {tlen} must be even")
    # A two-dimensional array to store the Fourier transform of the X vector
    # The first dimension is the time index
    # The second dimension is the frequency index
    # Number of frequency components
    kmax = int(maximum_frequency / total_bandwidth)
    nfreq = kmax + 1
    # Number of time components
    ntime = int(tlen / kmax)
    Xf = np.zeros((ntime, nfreq), dtype=complex)
    # Construct the wavelet
    time_array = np.arange(0, tlen) / sampling_rate
    wavelet = time_domain_wavelet(time_array, bandwidth_of_top_flat_region, total_bandwidth, sharpness)
    # Construct the X vector
    X = np.zeros((ntime, 2*kmax))
    Xf = np.zeros((ntime, nfreq), dtype=complex)
    X_kmax = int(tlen/2/kmax)
    for n in range(ntime):
        for j in range(2*kmax):
            for k in range(X_kmax):
                strain_t_index = (n*kmax + 2*k*kmax + j) % tlen
                wavelet_t_index = (2*k*kmax+j) % tlen
                X[n,j] += time_domain_strain[strain_t_index]*wavelet[wavelet_t_index]
            Xf[n] = np.conj(np.fft.rfft(X[n]))
    Xf = Xf * 2 * kmax
    # Extract the wavelet transform from Xf of the 0 frequency component
    output_0 = Xf[::2,0] / sampling_rate
    # Extract the wavelet transform from Xf of the kmax frequency components
    output_kmax = Xf[::2,kmax] / sampling_rate
    # Extract the wavelet transform from Xf of the other frequency components
    output_k = Xf[:,1:kmax] / sampling_rate * np.sqrt(2)
    for n in range(ntime):
        for k in range(1,kmax):
            if n+k % 2 == 0:
                output_k[n,k-1] = np.real(output_k[n,k-1])
            else:
                output_k[n,k-1] = -np.imag(output_k[n,k-1])
    return output_0, output_kmax, output_k