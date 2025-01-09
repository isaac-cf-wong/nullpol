import numpy as np
from tqdm import tqdm
from ..psd import simulate_psd_from_bilby_psd
from ..time_frequency_transform import (transform_wavelet_freq,
                                        transform_wavelet_freq_quadrature,
                                        get_shape_of_wavelet_transform)
from ..null_stream import time_shift
from ..detector import compute_whitened_time_frequency_domain_strain_array


def compute_sky_maximized_spectrogram(interferometers,
                                      frequency_domain_strain_array,
                                      wavelet_frequency_resolution,
                                      wavelet_nx,
                                      minimum_frequency,
                                      maximum_frequency,
                                      skypoints,
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
    energy_map_maximized = np.zeros((wavelet_Nt, wavelet_Nf))

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
                                                                                                              prefilter)
        time_frequency_domain_strain_array_time_shifted_quadrature = compute_whitened_time_frequency_domain_strain_array(time_frequency_domain_strain_array_time_shifted_quadrature,
                                                                                                                         psd_array,
                                                                                                                         prefilter)
        # Compute the energy map
        energy_map = (np.sum(np.abs(time_frequency_domain_strain_array_time_shifted)**2+np.abs(time_frequency_domain_strain_array_time_shifted_quadrature)**2, axis=0)) * 0.5
        energy_map_maximized = np.max((energy_map_maximized, energy_map), axis=0)
    return energy_map_maximized