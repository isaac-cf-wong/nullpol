import numpy as np
import scipy.stats


def compute_logistic_probability_map(whitened_time_frequency_spectrogram,
                                     confidence_threshold,
                                     df,
                                     time_frequency_filter,
                                     steepness=1.):
    probability_map = np.zeros_like(whitened_time_frequency_spectrogram, dtype=np.float64)
    threshold = scipy.stats.chi2.ppf(confidence_threshold, df)
    ntime, nfreq = probability_map.shape
    for i in range(ntime):
        for j in range(nfreq):
            if time_frequency_filter[i,j]:
                probability_map[i,j] = 1. / (1. + np.exp(-steepness * (whitened_time_frequency_spectrogram[i,j] - threshold)))
    return probability_map