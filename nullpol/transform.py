###################################################################
# Implementation of the Wilson-Dauechies time-frequency transform
#
# Reference:
# https://iopscience.iop.org/article/10.1088/1742-6596/363/1/012032/pdf
###################################################################
import numpy as np
from scipy.special import betainc
import scipy.integrate as integrate


def compute_frequency_domain_wavelet(frequency_array, bandwidth_of_top_flat_region, total_bandwidth, sharpness, dtype=float):
    """Compute frequency domain wavelet.

    Args:
        frequency_array (array-like): An array of frequencies
        bandwidth_of_top_flat_region (float): Bandwidth of the top flat region in Hz
        total_bandwidth (float): Total bandwidth in Hz
        sharpness (float): Sharpness of the wavelet

    Returns:
        numpy array: An array of the frequency domain wavelet
    """
    output = np.zeros_like(frequency_array, dtype=dtype)
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

def compute_time_domain_wavelet(time_array, bandwidth_of_top_flat_region, total_bandwidth, sharpness, precision=6):
    delta_t = time_array[1] - time_array[0]
    frequency_array = np.fft.rfftfreq(len(time_array), delta_t)
    frequency_domain_wavelet_array = compute_frequency_domain_wavelet(frequency_array, bandwidth_of_top_flat_region, total_bandwidth, sharpness)
    time_domain_wavelet_array = np.fft.irfft(frequency_domain_wavelet_array)
    # Normalize the time domain wavelet
    # The sum of the wavelet squared is 1.
    time_domain_wavelet_array *= np.sqrt(2 * np.pi / delta_t)
    # Find the safe truncation length
    tlen = len(time_domain_wavelet_array)
    energy = np.cumsum(time_domain_wavelet_array[:tlen//2]**2)
    idx = np.argmax(-np.log10(1 - energy/energy[-1])>=precision)
    return time_domain_wavelet_array, idx

def __compute_Xn(time_domain_data,
                 time_domain_wavelet,
                 n_idx,
                 M,
                 wavelet_time_idx_cutoff):
    """Compute the Xn vector.

    Args:
        time_domain_data (array-like): Array of the time domain data
        time_domain_wavelet (array-like): Array of the time domain wavelet
        n_idx (int): Time index of the Xn vector
        M (int): Number of frequency bins in the time-frequency representation
        wavelet_time_idx_cutoff (int): Time index of the cutoff of the wavelet

    Returns:
        _type_: _description_
    """
    M_times_2 = M*2
    output = np.zeros(M_times_2, dtype=time_domain_data.dtype)
    tlen_data = len(time_domain_data)
    tlen_wavelet = len(time_domain_wavelet)
    for i in range(M_times_2):
        # Find the minimum k
        k_min = int((-wavelet_time_idx_cutoff-i)/M_times_2)
        k_max = int((wavelet_time_idx_cutoff+1-i)/M_times_2)
        for k in range(k_min, k_max):
            wavelet_idx = k*M_times_2+i
            data_idx = wavelet_idx + n_idx*M
            output[i] += time_domain_data[data_idx%tlen_data]*time_domain_wavelet[wavelet_idx%tlen_wavelet]
    return output


def compute_time_frequency_transform_c(time_domain_data,
                                       sampling_rate,
                                       bandwidth_of_top_flat_region,
                                       total_bandwidth,
                                       sharpness):
    tlen = len(time_domain_data)
    if tlen%2 != 0:
        raise ValueError(f'Length of time_domain_data = {tlen} has to be even')
    # The maximum frequency
    fhigh = sampling_rate / 2
    # Number of frequency bins
    M = int(fhigh / total_bandwidth)
    if M%2 != 0:
        raise ValueError(f'sampling_rate / 2 divided by the total_bandwith is {M} has to be even')
    # Number of frequency indices
    M_length = M+1
    # Number of time indices
    n_length = int(tlen / M)
    # Construct the time domain wavelet
    time_array = np.arange(0, tlen) / sampling_rate
    time_domain_wavelet, wavelet_time_idx_cutoff = compute_time_domain_wavelet(time_array,
                                                                               bandwidth_of_top_flat_region,
                                                                               total_bandwidth,
                                                                               sharpness,
                                                                               precision=6)
    output_0 = np.zeros(n_length)
    output_M = np.zeros(n_length)
    output_n = np.zeros((n_length, M_length))
    for i in range(n_length):
        # Compute the Xn vector
        Xn = __compute_Xn(time_domain_data,
                          time_domain_wavelet,
                          i,
                          M,
                          wavelet_time_idx_cutoff)
        # Perform the inverse transform
        Xnf = np.conj(np.fft.rfft(Xn))
        # Copy the DC component
        output_0[i] = np.real(Xnf[0])
        output_M[i] = np.real(Xnf[M])
        # Copy the other components
        for m in range(1,M):
            if (m+i)%2 == 0:
                output_n[i, m] = np.real(Xnf[m])
            else:
                output_n[i, m] = -np.imag(Xnf[m])
    # Multiply output by the scaling
    scaling = np.sqrt(M) / sampling_rate
    output_0 *= scaling
    output_M *= scaling
    output_n *= (scaling * np.sqrt(2))
    # Compute the TF sampling times and TF sampling frequencies
    tf_sampling_times = np.arange(n_length) * M / sampling_rate
    tf_sampling_frequencies = np.arange(M_length) * total_bandwidth
    return output_0, output_M, output_n, tf_sampling_times, tf_sampling_frequencies

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
    wavelet, t_cutoff = compute_time_domain_wavelet(time_array, bandwidth_of_top_flat_region, total_bandwidth, sharpness)
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

def compute_time_delay_filter(time_delay,
                              tlen,
                              sampling_rate,
                              bandwidth_of_top_flat_region,
                              total_bandwidth,
                              sharpness):
    def compute_frequency_domain_wavelet_single_frequency(f, bandwidth_of_top_flat_region, total_bandwidth, sharpness):
        scaling = 1 / np.sqrt(total_bandwidth * 2 * np.pi)
        f_abs = np.abs(f)
        if f_abs < bandwidth_of_top_flat_region/2:
            return scaling
        elif f_abs <= total_bandwidth - bandwidth_of_top_flat_region/2:
            half_bandwidth_of_top_flat_region = bandwidth_of_top_flat_region/2
            return scaling * np.cos(betainc(sharpness, sharpness, (f_abs-half_bandwidth_of_top_flat_region)/(total_bandwidth-bandwidth_of_top_flat_region))*np.pi/2)
        else:
            return 0.
        
    def compute_T1l_integrand(f, bandwidth_of_top_flat_region, total_bandwidth, sharpness,
                             M, l, delta_t, time_delay):
        fwavelet = compute_frequency_domain_wavelet_single_frequency(f, bandwidth_of_top_flat_region, total_bandwidth, sharpness)
        return np.exp(1.j*2*np.pi*f*(M*l*delta_t + time_delay))*fwavelet**2

    def compute_T2l_integrand(f, bandwidth_of_top_flat_region, total_bandwidth, sharpness,
                              M, l, delta_t, time_delay, domega):
        # Since fwavelet is real, do not need to take the conjugate of fwavelet1.
        fwavelet1 = compute_frequency_domain_wavelet_single_frequency(f-domega/2, bandwidth_of_top_flat_region, total_bandwidth, sharpness)
        fwavelet2 = compute_frequency_domain_wavelet_single_frequency(f+domega/2, bandwidth_of_top_flat_region, total_bandwidth, sharpness)
        return np.exp(1.j*2*np.pi*f*(M*l*delta_t + time_delay))*fwavelet1*fwavelet2

    if tlen%2 != 0:
        raise ValueError(f'Length of time_domain_data = {tlen} has to be even')
    # The maximum frequency
    fhigh = sampling_rate / 2
    # The sampling interval
    delta_t = 1 / sampling_rate
    # Number of frequency bins
    M = int(fhigh / total_bandwidth)
    if M%2 != 0:
        raise ValueError(f'sampling_rate / 2 divided by the total_bandwith is {M} has to be even')
    # Number of frequency indices
    M_length = M+1
    # Number of time indices
    n_length = int(tlen / M)
    # Compute the domega
    domega = 2 * np.pi * total_bandwidth
    # Compute the frequency array
    frequency_array = np.fft.rfftfreq(tlen, delta_t)
    # Compute the wavelet
    fwavelet = compute_frequency_domain_wavelet(frequency_array, bandwidth_of_top_flat_region, total_bandwidth, sharpness, dtype=float)
    # Compute the wavelet squared
    fwavelet2 = fwavelet**2
    # The output filter
    output = np.zeros((n_length, M_length, n_length, M_length))
    # Upper bound of the integral
    fhigh = total_bandwidth - bandwidth_of_top_flat_region/2
    # Lower bound of the integral
    flow = -fhigh    
    for n1 in range(n_length):
        for n2 in range(n_length):
            for m in range(M_length):
                l = n2 - n1
                # Compute the T integral for the 0th component
                T1l = integrate.quad(compute_T1l_integrand, flow, fhigh,
                                     args=(bandwidth_of_top_flat_region,
                                           total_bandwidth,
                                           sharpness,
                                           M, l,
                                           delta_t,
                                           time_delay),
                                           complex_func=True)[0]*2*np.pi
                # Compute the T integral for the +-1 th components
                T2l = integrate.quad(compute_T2l_integrand, flow, fhigh,
                                     args=(bandwidth_of_top_flat_region,
                                           total_bandwidth,
                                           sharpness,
                                           M, l,
                                           delta_t,
                                           time_delay,
                                           domega),
                                           complex_func=True)[0]*2*np.pi
                if l%2 == 0:
                    Cl = 1
                    Clp1 = 1.j
                else:
                    Cl = 1.j
                    Clp1 = 1
                # The 0th component
                output[n1, m, n2, m] = (-1)**(l*n1)*np.real(np.conj(Cl)*np.exp(1.j*m*domega*time_delay)*T1l)
                # The -1th component
                output[n1, m, n2, m-1] = (-1)**(n1+m)*(-1)**(l*n1) * np.real(np.conj(Clp1)*np.exp(1.j*(m-0.5)*domega*time_delay)*(-1.j)**(l)*T2l)
                # The +1th component
                output[n1, m, n2, (m+1)%M_length] = (-1)**(n1+m)*(-1)**(l*n1) * np.real(np.conj(Clp1)*np.exp(1.j*(m+0.5)*domega*time_delay)*(+1.j)**(l)*T2l)
                if m == 0 or m == M:
                    output[n1, m, n2, m-1] *= np.sqrt(2)
                    output[n1, m, n2, (m+1)%M_length] *= np.sqrt(2)
                elif m+1 == M:
                    output[n1, m, n2, m+1] *= np.sqrt(2)
                elif m-1 == M:
                    output[n1, m, n2, m-1] *= np.sqrt(2)
    return output

def apply_time_delay_filter(wnm, time_delay_filter):
    return np.einsum('lk,nmlk->nm', wnm, time_delay_filter)