import numpy as np
from tqdm import tqdm
from ..utility import logger
from ..psd import simulate_psd_from_bilby_psd
from ..null_stream import (time_shift,
                           compute_whitened_time_frequency_domain_strain_array)
from ..time_frequency_transform import (transform_wavelet_freq,
                                        transform_wavelet_freq_quadrature,
                                        get_shape_of_wavelet_transform)
from .single import clustering
from .plot import *


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
                                  simulate_psd_nsample=1000):
    # Simulate the wavelet PSDs
    if psd_array is None:
        psd_array = np.array([simulate_psd_from_bilby_psd(psd=ifo.power_spectral_density,
                                                        seglen=ifo.duration,
                                                        srate=ifo.sampling_frequency,
                                                        wavelet_frequency_resolution=wavelet_frequency_resolution,
                                                        nsample=simulate_psd_nsample,
                                                        nx=wavelet_nx) for ifo in interferometers])
    # Draw random sky points
    ra_array = np.random.uniform(0, 2 * np.pi, size=skypoints)
    dec_array = np.arcsin(np.random.uniform(-1, 1, size=skypoints))
    geocent_time = interferometers[0].start_time + interferometers[0].duration / 2
    wavelet_Nt, wavelet_Nf = get_shape_of_wavelet_transform(duration=interferometers[0].duration,
                                                            sampling_frequency=interferometers[0].sampling_frequency,
                                                            wavelet_frequency_resolution=wavelet_frequency_resolution)
    prefilter = np.full((wavelet_Nt, wavelet_Nf), True)
    # Remove the components beyond the frequency range

    if minimum_frequency is not None:
        freq_low_idx = int(np.ceil(minimum_frequency / wavelet_frequency_resolution))
        prefilter[:,:freq_low_idx] = False

    if maximum_frequency is not None:
        freq_high_idx = int(np.floor(maximum_frequency / wavelet_frequency_resolution))
        prefilter[:,freq_high_idx:] = False
    energy_map_combined = np.zeros((wavelet_Nt, wavelet_Nf))

    for i in tqdm(range(skypoints), desc='Generating energy map'):
        # Time shift the data
        frequency_domain_strain_array_time_shifted = time_shift(interferometers=interferometers,
                                                                ra=ra_array[i],
                                                                dec=dec_array[i],
                                                                gps_time=geocent_time,
                                                                strain_data_array=frequency_domain_strain_array)
        # Transform the time-shifted data to the time-frequency domain
        time_frequency_domain_strain_array_time_shifted = np.array([transform_wavelet_freq(data,
                                                                                           wavelet_Nf,
                                                                                           wavelet_Nt,
                                                                                           wavelet_nx) for data in frequency_domain_strain_array_time_shifted])
        time_frequency_domain_strain_array_time_shifted_quadrature = np.array([transform_wavelet_freq_quadrature(data,
                                                                                                                 wavelet_Nf,
                                                                                                                 wavelet_Nt,
                                                                                                                 wavelet_nx) for data in frequency_domain_strain_array_time_shifted])
        # Whiten the time-frequency data
        time_frequency_domain_strain_array_time_shifted = compute_whitened_time_frequency_domain_strain_array(time_frequency_domain_strain_array_time_shifted,
                                                                                                              psd_array,
                                                                                                              prefilter,
                                                                                                              interferometers[0].sampling_frequency)
        time_frequency_domain_strain_array_time_shifted_quadrature = compute_whitened_time_frequency_domain_strain_array(time_frequency_domain_strain_array_time_shifted_quadrature,
                                                                                                                         psd_array,
                                                                                                                         prefilter,
                                                                                                                         interferometers[0].sampling_frequency)
        # Compute the energy map
        energy_map = np.sum(np.abs(time_frequency_domain_strain_array_time_shifted)**2+np.abs(time_frequency_domain_strain_array_time_shifted_quadrature)**2, axis=0)
        energy_map_combined += energy_map
    energy_threshold = np.quantile(energy_map_combined[energy_map_combined>0.], threshold)
    energy_filter = energy_map_combined > energy_threshold
    dt = interferometers[0].duration / wavelet_Nt
    output = clustering(energy_filter, dt, wavelet_frequency_resolution, padding_time=time_padding, padding_freq=frequency_padding)
    # Clean the filter again
    output = output.astype(bool)

    if minimum_frequency is not None:
        output[:,:freq_low_idx] = False

    if maximum_frequency is not None:
        output[:,freq_high_idx:] = False

    if return_sky_maximized_spectrogram:
        return output.astype(bool), energy_map_combined
    else:
        return output.astype(bool)

def write_time_frequency_filter(filename, time_frequency_filter):
    np.save(filename, time_frequency_filter)
    logger.info(f"Time-frequency filter saved to {filename}")
