from __future__ import annotations

import numpy as np
from tqdm import tqdm

from ..detector import compute_whitened_frequency_domain_strain_array
from ..null_stream import compute_time_shifted_frequency_domain_strain_array
from ..time_frequency_transform import (get_shape_of_wavelet_transform,
                                        transform_wavelet_freq,
                                        transform_wavelet_freq_quadrature)


def compute_sky_maximized_spectrogram(interferometers,
                                      frequency_domain_strain_array,
                                      wavelet_frequency_resolution,
                                      wavelet_nx,
                                      skypoints):
    # Whiten the data
    psd_array = np.array([
        ifo.power_spectral_density_array for ifo in interferometers
    ])
    whitened_frequency_domain_strain_array = \
        compute_whitened_frequency_domain_strain_array(
            frequency_mask=interferometers[0].frequency_mask,
            frequency_resolution=1./interferometers[0].duration,
            frequency_domain_strain_array=frequency_domain_strain_array,
            power_spectral_density_array=psd_array
        )
    # Draw random sky points
    ra_array = np.random.uniform(0, 2 * np.pi, size=skypoints)
    dec_array = np.arcsin(np.random.uniform(-1, 1, size=skypoints))
    geocent_time = interferometers[0].start_time + interferometers[0].duration / 2
    wavelet_Nt, wavelet_Nf = get_shape_of_wavelet_transform(
        duration=interferometers[0].duration,
        sampling_frequency=interferometers[0].sampling_frequency,
        wavelet_frequency_resolution=wavelet_frequency_resolution)
    energy_map_maximized = np.zeros((wavelet_Nt, wavelet_Nf))

    for i in tqdm(range(skypoints), desc='Generating energy map'):
        # Compute the time delays
        time_delay_array = np.array([
            ifo.time_delay_from_geocenter(
                ra=ra_array[i],
                dec=dec_array[i],
                time=geocent_time
            ) for ifo in interferometers
        ])
        # Time shift the data
        frequency_domain_strain_array_time_shifted = compute_time_shifted_frequency_domain_strain_array(
            frequency_array=interferometers[0].frequency_array,
            frequency_mask=interferometers[0].frequency_mask,
            frequency_domain_strain_array=whitened_frequency_domain_strain_array,
            time_delay_array=time_delay_array
        )
        # Transform the time-shifted data to the time-frequency domain
        time_frequency_domain_strain_array_time_shifted = np.array([transform_wavelet_freq(
            data,
            interferometers[0].sampling_frequency,
            wavelet_frequency_resolution,
            wavelet_nx) for data in frequency_domain_strain_array_time_shifted])
        time_frequency_domain_strain_array_time_shifted_quadrature = np.array([transform_wavelet_freq_quadrature(
            data,
            interferometers[0].sampling_frequency,
            wavelet_frequency_resolution,
            wavelet_nx) for data in frequency_domain_strain_array_time_shifted])
        # Compute the energy map
        energy_map = (np.sum(np.abs(time_frequency_domain_strain_array_time_shifted)**2+np.abs(time_frequency_domain_strain_array_time_shifted_quadrature)**2, axis=0)) * 0.5
        energy_map_maximized = np.max((energy_map_maximized, energy_map),
                                      axis=0)
    return energy_map_maximized
