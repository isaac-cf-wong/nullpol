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

def time_frequency_transform(time_domain_strain):
    pass