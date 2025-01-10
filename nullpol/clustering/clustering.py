import numpy as np
import scipy.stats
from .sky_maximized_spectrogram import compute_sky_maximized_spectrogram
from .single import clustering
from ..time_frequency_transform import get_shape_of_wavelet_transform
from ..utility import logger


def run_time_frequency_clustering(interferometers,
                                  frequency_domain_strain_array,
                                  wavelet_frequency_resolution,
                                  wavelet_nx,
                                  minimum_frequency,
                                  maximum_frequency,
                                  threshold,
                                  time_padding,
                                  frequency_padding,
                                  skypoints,
                                  return_sky_maximized_spectrogram=False,
                                  psd_array=None,
                                  simulate_psd_nsample=1000,
                                  threshold_type="quantile"):
    sky_maximized_spectrogram = compute_sky_maximized_spectrogram(interferometers=interferometers,
                                                                  frequency_domain_strain_array=frequency_domain_strain_array,
                                                                  wavelet_frequency_resolution=wavelet_frequency_resolution,
                                                                  wavelet_nx=wavelet_nx,
                                                                  minimum_frequency=minimum_frequency,
                                                                  maximum_frequency=maximum_frequency,
                                                                  skypoints=skypoints,
                                                                  psd_array=psd_array,
                                                                  simulate_psd_nsample=simulate_psd_nsample)
    if threshold_type == "quantile":
        energy_threshold = np.quantile(sky_maximized_spectrogram[sky_maximized_spectrogram>0.], threshold)
    elif threshold_type == "confidence":
        energy_threshold = scipy.stats.chi2.ppf(threshold, df=len(interferometers))
    energy_filter = sky_maximized_spectrogram > energy_threshold
    if np.all(~energy_filter):
        logger.warning(f"No time-frequency pixel passes the energy threshold = {energy_threshold}.")
        logger.warning("Switched to quantile threshold.")
        energy_threshold = np.quantile(sky_maximized_spectrogram[sky_maximized_spectrogram>0.], threshold)
        energy_filter = sky_maximized_spectrogram > energy_threshold
    wavelet_Nt, wavelet_Nf = get_shape_of_wavelet_transform(duration=interferometers[0].duration,
                                                            sampling_frequency=interferometers[0].sampling_frequency,
                                                            wavelet_frequency_resolution=wavelet_frequency_resolution)
    dt = interferometers[0].duration / wavelet_Nt
    output = clustering(energy_filter, dt, wavelet_frequency_resolution, padding_time=time_padding, padding_freq=frequency_padding)
    # Clean the filter to ensure no leakage.
    output = output.astype(np.float64)    
    if minimum_frequency is not None:
        freq_low_idx = int(np.ceil(minimum_frequency / wavelet_frequency_resolution))
        output[:,:freq_low_idx] = 0.

    if maximum_frequency is not None:
        freq_high_idx = int(np.floor(maximum_frequency / wavelet_frequency_resolution))
        output[:,freq_high_idx:] = 0.

    if return_sky_maximized_spectrogram:
        return output, sky_maximized_spectrogram
    else:
        return output