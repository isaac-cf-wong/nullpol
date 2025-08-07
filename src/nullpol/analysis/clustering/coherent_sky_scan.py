from __future__ import annotations

import numpy as np
from tqdm import tqdm

from ..null_stream import compute_time_shifted_frequency_domain_strain_array
from ..signal_processing import compute_whitened_frequency_domain_strain_array
from ..tf_transforms import (
    get_shape_of_wavelet_transform,
    transform_wavelet_freq,
    transform_wavelet_freq_quadrature,
)


def scan_sky_for_coherent_power(
    interferometers, frequency_domain_strain_array, wavelet_frequency_resolution, wavelet_nx, skypoints
):
    """Scan sky positions to find coherent power across detector network.

    Generates a time-frequency spectrogram by maximizing over different sky positions
    to identify the most coherent excess power across the detector network. This is
    used to find candidate gravitational wave signals without knowing the source location.

    The algorithm:
    1. Whitens the frequency domain strain data using detector PSDs
    2. For each random sky position, computes time delays between detectors
    3. Time-shifts data to account for light travel time differences
    4. Transforms time-shifted data to time-frequency domain using wavelets
    5. Computes power spectrogram for each sky position
    6. Returns pixel-wise maximum across all sky positions

    Args:
        interferometers (list): List of bilby.gw.detector.Interferometer objects
            containing strain data and detector parameters.
        frequency_domain_strain_array (numpy.ndarray): Frequency domain strain data
            with shape (n_detectors, n_frequencies).
        wavelet_frequency_resolution (float): Frequency resolution for wavelet transform in Hz.
        wavelet_nx (float): Wavelet steepness parameter controlling time-frequency localization.
        skypoints (int): Number of random sky positions to test.

    Returns:
        numpy.ndarray: Sky-maximized spectrogram with shape (n_time, n_frequency).
            Each pixel contains the maximum power across all tested sky positions.

    Note:
        - Sky positions are randomly sampled uniformly over the sphere
        - Uses both real and quadrature wavelet components for power calculation
        - The returned spectrogram represents coherent power maximized over sky location
        - Whitening uses the average PSD over the data duration
    """
    # Whiten the data
    psd_array = np.array([ifo.power_spectral_density_array for ifo in interferometers])
    whitened_frequency_domain_strain_array = compute_whitened_frequency_domain_strain_array(
        frequency_mask=interferometers[0].frequency_mask,
        frequency_resolution=1.0 / interferometers[0].duration,
        frequency_domain_strain_array=frequency_domain_strain_array,
        power_spectral_density_array=psd_array,
    )
    # Draw random sky points
    ra_array = np.random.uniform(0, 2 * np.pi, size=skypoints)
    dec_array = np.arcsin(np.random.uniform(-1, 1, size=skypoints))
    geocent_time = interferometers[0].start_time + interferometers[0].duration / 2
    wavelet_Nt, wavelet_Nf = get_shape_of_wavelet_transform(
        duration=interferometers[0].duration,
        sampling_frequency=interferometers[0].sampling_frequency,
        wavelet_frequency_resolution=wavelet_frequency_resolution,
    )
    energy_map_maximized = np.zeros((wavelet_Nt, wavelet_Nf))

    for i in tqdm(range(skypoints), desc="Generating energy map"):
        # Compute the time delays
        time_delay_array = np.array(
            [
                ifo.time_delay_from_geocenter(ra=ra_array[i], dec=dec_array[i], time=geocent_time)
                for ifo in interferometers
            ]
        )
        # Time shift the data
        frequency_domain_strain_array_time_shifted = compute_time_shifted_frequency_domain_strain_array(
            frequency_array=interferometers[0].frequency_array,
            frequency_mask=interferometers[0].frequency_mask,
            frequency_domain_strain_array=whitened_frequency_domain_strain_array,
            time_delay_array=time_delay_array,
        )
        # Transform the time-shifted data to the time-frequency domain
        time_frequency_domain_strain_array_time_shifted = np.array(
            [
                transform_wavelet_freq(
                    data, interferometers[0].sampling_frequency, wavelet_frequency_resolution, wavelet_nx
                )
                for data in frequency_domain_strain_array_time_shifted
            ]
        )
        time_frequency_domain_strain_array_time_shifted_quadrature = np.array(
            [
                transform_wavelet_freq_quadrature(
                    data, interferometers[0].sampling_frequency, wavelet_frequency_resolution, wavelet_nx
                )
                for data in frequency_domain_strain_array_time_shifted
            ]
        )
        # Compute the energy map
        energy_map = (
            np.sum(
                np.abs(time_frequency_domain_strain_array_time_shifted) ** 2
                + np.abs(time_frequency_domain_strain_array_time_shifted_quadrature) ** 2,
                axis=0,
            )
        ) * 0.5
        energy_map_maximized = np.max((energy_map_maximized, energy_map), axis=0)
    return energy_map_maximized
