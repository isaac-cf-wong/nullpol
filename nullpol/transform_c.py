import numpy as np
import scipy.special

def get_phif(f, df, A, B, sharpness):
    f_abs = abs(f)
    output = np.zeros(len(f))
    output[f_abs < A] = 1. / np.sqrt(2. * np.pi * df)
    f_idx = (f_abs >= A) & (f_abs < A + B)
    output[f_idx] = np.cos(np.pi / 2 * scipy.special.betainc(sharpness, sharpness, (f_abs[f_idx] - A) / B)) / np.sqrt(2. * np.pi * df)
    return output

def get_phit(length, fs, M, K, sharpness, precision):
    fhigh = fs / 2
    df = fhigh / M
    A = fhigh /2. * (1. / M - 1. / K)
    B = fhigh / K
    df_phi = fs / length
    flength = length // 2 + 1
    sampling_frequencies = np.arange(flength) * df_phi
    
    # Get the frequency domain wavelet
    phif = get_phif(sampling_frequencies, df_phi, A, B, sharpness)
    phit = np.fft.irfft(phif)

    # Normalization constant
    norm = np.sqrt(fs * 2 * np.pi)
    phit *= norm

    # Maximum length
    max_length = (length - 1) // 2
    # Calculate the total energy
    return phit