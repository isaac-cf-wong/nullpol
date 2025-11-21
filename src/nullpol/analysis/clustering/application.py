from __future__ import annotations

import numpy as np
import scipy.stats

from ..tf_transforms import get_shape_of_wavelet_transform
from ...utils import NullpolError, logger
from .algorithm import clustering
from .coherent_sky_scan import scan_sky_for_coherent_power


def run_time_frequency_clustering(
    interferometers,
    frequency_domain_strain_array,
    wavelet_frequency_resolution,
    wavelet_nx,
    threshold,
    time_padding,
    frequency_padding,
    skypoints,
    return_sky_maximized_spectrogram=False,
    threshold_type="quantile",
):
    """Perform time-frequency clustering analysis on interferometer data.

    Identifies significant excess power regions in time-frequency spectrograms by:
    1. Computing sky-maximized spectrograms across multiple sky positions
    2. Applying statistical thresholds to identify candidate pixels
    3. Clustering connected pixels and selecting the largest cluster
    4. Applying padding around the cluster for robust analysis

    Args:
        interferometers (list): List of bilby.gw.detector.Interferometer objects.
        frequency_domain_strain_array (numpy.ndarray): Frequency domain strain data
            with shape (n_detectors, n_frequencies).
        wavelet_frequency_resolution (float): Frequency resolution for wavelet transform in Hz.
        wavelet_nx (float): Wavelet steepness parameter.
        threshold (float): Threshold value for pixel selection. Interpretation depends
            on threshold_type.
        time_padding (float): Time padding around clusters in seconds.
        frequency_padding (float): Frequency padding around clusters in Hz.
        skypoints (int): Number of random sky positions to test.
        return_sky_maximized_spectrogram (bool, optional): If True, also return the
            sky-maximized spectrogram. Defaults to False.
        threshold_type (str, optional): Method for interpreting threshold value:
            - "quantile": threshold is quantile of non-zero values (0-1)
            - "confidence": threshold is confidence level for chi-squared distribution
            - "variance": threshold multiplied by number of detectors
            Defaults to "quantile".

    Returns:
        numpy.ndarray or tuple: If return_sky_maximized_spectrogram is False,
            returns cluster mask array with shape (n_time, n_frequency).
            If True, returns tuple of (cluster_mask, sky_maximized_spectrogram).

    Raises:
        NullpolError: If threshold_type is not recognized.

    Note:
        The cluster is cleaned to remove contributions outside the interferometer
        frequency band [minimum_frequency, maximum_frequency].
    """
    sky_maximized_spectrogram = scan_sky_for_coherent_power(
        interferometers=interferometers,
        frequency_domain_strain_array=frequency_domain_strain_array,
        wavelet_frequency_resolution=wavelet_frequency_resolution,
        wavelet_nx=wavelet_nx,
        skypoints=skypoints,
    )
    if threshold_type == "quantile":
        energy_threshold = np.quantile(sky_maximized_spectrogram[sky_maximized_spectrogram > 0.0], threshold)
    elif threshold_type == "confidence":
        energy_threshold = scipy.stats.chi2.ppf(threshold, df=len(interferometers))
    elif threshold_type == "variance":
        # Compute the median along the time axis
        energy_threshold = threshold * len(interferometers)
    else:
        raise NullpolError(f"threshold_type={threshold_type} is not recognized.")
    energy_filter = sky_maximized_spectrogram > energy_threshold
    if np.all(~energy_filter):
        logger.warning(f"No time-frequency pixel passes the energy threshold = {energy_threshold}.")
        logger.warning("Returning an empty cluster filter.")
        output = energy_filter.astype(np.float64)
    else:
        wavelet_Nt, _wavelet_Nf = get_shape_of_wavelet_transform(
            duration=interferometers[0].duration,
            sampling_frequency=interferometers[0].sampling_frequency,
            wavelet_frequency_resolution=wavelet_frequency_resolution,
        )
        dt = interferometers[0].duration / wavelet_Nt
        output = clustering(
            energy_filter, dt, wavelet_frequency_resolution, padding_time=time_padding, padding_freq=frequency_padding
        )
        # Clean the filter to ensure no leakage.
        output = output.astype(np.float64)
        freq_low_idx = int(np.ceil(interferometers[0].minimum_frequency / wavelet_frequency_resolution))
        output[:, :freq_low_idx] = 0.0

        freq_high_idx = int(np.floor(interferometers[0].maximum_frequency / wavelet_frequency_resolution))
        output[:, freq_high_idx:] = 0.0

    if return_sky_maximized_spectrogram:
        return output, sky_maximized_spectrogram
    return output
