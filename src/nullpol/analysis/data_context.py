"""Time-frequency data context for gravitational wave analysis.

This module provides centralized data management and preprocessing capabilities for
time-frequency domain gravitational wave analysis. The TimeFrequencyDataContext class
serves as the primary interface for handling interferometer data, frequency arrays,
whitening operations, and wavelet transforms in a unified, efficient manner.

Key Features:
    - Interferometer data validation and management
    - Frequency domain strain processing and whitening
    - Time-shift corrections for multi-detector alignment
    - Wavelet transform integration for time-frequency analysis
    - Time-frequency filtering and masking operations
    - Optimized numerical computations using Numba JIT compilation

The module includes high-performance Numba-optimized functions for computationally
intensive operations such as whitening, time-shifting, and array manipulations,
ensuring efficient processing of large gravitational wave datasets.
"""

from __future__ import annotations

from typing import Optional

import bilby
import numpy as np
from numba import njit

from .tf_transforms import get_shape_of_wavelet_transform
from ..utils import logger


# =============================================================================
# SIGNAL PROCESSING FUNCTIONS (NUMBA OPTIMIZED)
# =============================================================================


@njit
def compute_whitened_frequency_domain_strain_array(
    frequency_mask,
    frequency_resolution,
    frequency_domain_strain_array,
    power_spectral_density_array,
):
    """Compute the whitened frequency domain strain array.

    Args:
        frequency_mask (numpy array): A boolean array of frequency mask.
        frequency_resolution (float): Frequency resolution in Hz.
        frequency_domain_strain_array (numpy array): Frequency domain strain array (detector, frequency).
        power_spectral_density_array (numpy array): Power spectral density array (detector, frequency).

    Returns:
        numpy: Whitened frequency domain strain array.
    """
    output = np.zeros_like(frequency_domain_strain_array)
    output[:, frequency_mask] = frequency_domain_strain_array[:, frequency_mask] / np.sqrt(
        power_spectral_density_array[:, frequency_mask] / (2 * frequency_resolution)
    )
    return output


@njit
def compute_time_shifted_frequency_domain_strain(frequency_array, frequency_mask, frequency_domain_strain, time_delay):
    """Apply time shift to frequency domain strain data for a single detector.

    Shifts the strain data in time by applying a frequency-dependent phase
    factor in the frequency domain. This is equivalent to a time translation
    in the time domain but computed more efficiently in frequency space.

    The time shift is applied using the formula:
    h_shifted(f) = h(f) × exp(2πif×Δt)

    where h(f) is the original frequency domain strain, f is frequency,
    and Δt is the time delay.

    Args:
        frequency_array (numpy.ndarray): Frequency values with shape (n_frequencies,).
            Array containing the frequency bins in Hz.
        frequency_mask (numpy.ndarray): Boolean mask with shape (n_frequencies,)
            indicating which frequency bins to process. Time shift is applied
            only where the mask is True.
        frequency_domain_strain (numpy.ndarray): Complex frequency domain strain
            with shape (n_frequencies,). Input strain data to be time-shifted.
        time_delay (float): Time delay in seconds. Positive values shift the
            signal forward in time (earlier arrival), negative values shift
            backward (later arrival).

    Returns:
        numpy.ndarray: Time-shifted frequency domain strain with shape
            (n_frequencies,). Complex-valued array with the same shape as input.
            Unmasked frequencies are set to zero.

    Note:
        This function is compiled with Numba for performance. The time shift
        preserves the signal's spectral content while changing its phase
        according to the specified delay.
    """
    output = np.zeros_like(frequency_domain_strain)
    phase_shift = np.exp(1.0j * 2 * np.pi * frequency_array[frequency_mask] * time_delay)
    output[frequency_mask] = frequency_domain_strain[frequency_mask] * phase_shift
    return output


@njit
def compute_time_shifted_frequency_domain_strain_array(
    frequency_array, frequency_mask, frequency_domain_strain_array, time_delay_array
):
    """Apply time shifts to frequency domain strain data for multiple detectors.

    Applies different time shifts to strain data from multiple detectors
    simultaneously. This is commonly used to align multi-detector data to
    a common reference frame (e.g., geocenter) where each detector has a
    different arrival time delay.

    The time shift for each detector is applied using:
    h_shifted[i](f) = h[i](f) × exp(2πif×Δt[i])

    where h[i](f) is the frequency domain strain for detector i, f is frequency,
    and Δt[i] is the time delay for detector i.

    Args:
        frequency_array (numpy.ndarray): Frequency values with shape (n_frequencies,).
            Array containing the frequency bins in Hz.
        frequency_mask (numpy.ndarray): Boolean mask with shape (n_frequencies,)
            indicating which frequency bins to process. Time shifts are applied
            only where the mask is True.
        frequency_domain_strain_array (numpy.ndarray): Complex frequency domain
            strain data with shape (n_detectors, n_frequencies). Input strain
            data from multiple detectors to be time-shifted.
        time_delay_array (numpy.ndarray): Time delays with shape (n_detectors,).
            Array of time delays in seconds, one for each detector. Positive
            values shift signals forward in time, negative values shift backward.

    Returns:
        numpy.ndarray: Time-shifted frequency domain strain array with shape
            (n_detectors, n_frequencies). Complex-valued array where each row
            contains the time-shifted strain for one detector. Unmasked
            frequencies are set to zero.

    Note:
        This function is compiled with Numba for performance. It efficiently
        processes multiple detectors by using numpy's outer product to compute
        all phase shifts simultaneously, then applies them element-wise.
    """
    output = np.zeros_like(frequency_domain_strain_array)
    phase_shift_array = np.exp(np.outer(time_delay_array, 1.0j * 2 * np.pi * frequency_array[frequency_mask]))
    output[:, frequency_mask] = frequency_domain_strain_array[:, frequency_mask] * phase_shift_array
    return output


# pylint: disable=too-many-instance-attributes
class TimeFrequencyDataContext:
    """Centralized data management and preprocessing for time-frequency analysis.

    This class handles all data-related operations including interferometer data validation,
    frequency array management, whitening operations, and wavelet transforms. It provides
    a clean interface for accessing processed data without embedding business logic.

    Args:
        interferometers (list): List of interferometers.
        wavelet_frequency_resolution (float): The frequency resolution of the wavelet transform.
        wavelet_nx (int): The number of points in the wavelet transform.
        time_frequency_filter (np.ndarray, optional): The time-frequency filter.
    """

    def __init__(
        self,
        interferometers,
        wavelet_frequency_resolution,
        wavelet_nx,
        time_frequency_filter=None,
    ):
        # Load and validate interferometers
        self._interferometers = bilby.gw.detector.networks.InterferometerList(interferometers)
        self._validate_interferometers(self._interferometers)

        # Set up basic data properties
        self._duration = self.interferometers[0].duration
        self._sampling_frequency = self.interferometers[0].sampling_frequency
        self._frequency_domain_strain_array = np.array([ifo.frequency_domain_strain for ifo in self.interferometers])
        self._frequency_array = self.interferometers[0].frequency_array.copy()

        # Set up frequency mask (logical AND of all interferometer masks)
        self._frequency_mask = np.logical_and.reduce([ifo.frequency_mask for ifo in self.interferometers])
        if self._frequency_mask[-1]:
            self._frequency_mask[-1] = False
            logger.warning("The frequency mask at the Nyquist frequency is not False. It is set to be False.")

        self._masked_frequency_array = self._frequency_array[self._frequency_mask]
        self._frequency_resolution = self._frequency_array[1] - self._frequency_array[0]

        # Set up power spectral density array
        self._power_spectral_density_array = np.array(
            [ifo.power_spectral_density_array for ifo in self.interferometers]
        )

        # Wavelet transform parameters
        self._wavelet_frequency_resolution = wavelet_frequency_resolution
        self._wavelet_nx = wavelet_nx
        self._tf_Nt, self._tf_Nf = get_shape_of_wavelet_transform(
            self.duration, self.sampling_frequency, self.wavelet_frequency_resolution
        )

        # Time-frequency filter
        self._time_frequency_filter = time_frequency_filter

        # Validate filter if provided as array
        if isinstance(self._time_frequency_filter, np.ndarray):
            self._validate_time_frequency_filter()

        # Cached processed data
        self._whitened_frequency_domain_strain_array = None
        self._cached_whitened_frequency_domain_strain_array_at_geocenter = None

    @property
    def interferometers(self) -> bilby.gw.detector.networks.InterferometerList:
        """A list of interferometers."""
        return self._interferometers

    @property
    def duration(self) -> float:
        """Duration of strain data in seconds."""
        return self._duration

    @property
    def sampling_frequency(self) -> float:
        """Sampling frequency of strain data in Hz."""
        return self._sampling_frequency

    @property
    def frequency_domain_strain_array(self) -> np.ndarray:
        """An array of frequency domain strain of interferometers.

        Returns:
            numpy.ndarray: Frequency domain strain array (detector, frequency).
        """
        return self._frequency_domain_strain_array

    @property
    def frequency_array(self) -> np.ndarray:
        """Frequency array."""
        return self._frequency_array

    @property
    def frequency_mask(self) -> np.ndarray:
        """Frequency mask.

        Returns:
            numpy.ndarray: A boolean array of frequency mask.
        """
        return self._frequency_mask

    @property
    def masked_frequency_array(self) -> np.ndarray:
        """Masked frequency array."""
        return self._masked_frequency_array

    @property
    def frequency_resolution(self) -> float:
        """Frequency resolution in Hz."""
        return self._frequency_resolution

    @property
    def power_spectral_density_array(self) -> np.ndarray:
        """Power spectral density array of interferometers."""
        return self._power_spectral_density_array

    @property
    def wavelet_frequency_resolution(self) -> float:
        """Frequency resolution in wavelet domain in Hz."""
        return self._wavelet_frequency_resolution

    @property
    def wavelet_nx(self) -> int:
        """Steepness of the filter in wavelet transform."""
        return self._wavelet_nx

    @property
    def tf_Nt(self) -> int:
        """Number of time bins in the wavelet domain."""
        return self._tf_Nt

    @property
    def tf_Nf(self) -> int:
        """Number of frequency bins in the wavelet domain."""
        return self._tf_Nf

    @property
    def time_frequency_filter(self) -> Optional[np.ndarray]:
        """Time frequency filter.

        Returns:
            Optional[numpy.ndarray]: Time frequency filter (time, frequency).
        """
        if isinstance(self._time_frequency_filter, str):
            self._time_frequency_filter = np.load(self._time_frequency_filter)
        return self._time_frequency_filter

    @property
    def whitened_frequency_domain_strain_array(self) -> Optional[np.ndarray]:
        """Whitened frequency domain strain array of the interferometers.

        Returns:
            Optional[numpy.ndarray]: Whitened frequency domain strain array (detector, frequency).
        """
        if (
            self._whitened_frequency_domain_strain_array is None
            and self.frequency_domain_strain_array is not None
            and self.power_spectral_density_array is not None
        ):
            self._whitened_frequency_domain_strain_array = compute_whitened_frequency_domain_strain_array(
                frequency_mask=self.frequency_mask,
                frequency_resolution=self.frequency_resolution,
                frequency_domain_strain_array=self.frequency_domain_strain_array,
                power_spectral_density_array=self.power_spectral_density_array,
            )
        return self._whitened_frequency_domain_strain_array

    def compute_time_delay_array(self, parameters: dict) -> np.ndarray:
        """Compute an array of time delays for the given sky position and time.

        Args:
            parameters (dict): Dictionary containing 'ra', 'dec', and 'geocent_time'.

        Returns:
            numpy.ndarray: Time delay array.
        """
        return np.array(
            [
                ifo.time_delay_from_geocenter(
                    ra=parameters["ra"], dec=parameters["dec"], time=parameters["geocent_time"]
                )
                for ifo in self.interferometers
            ]
        )

    def compute_whitened_strain_at_geocenter(self, parameters: dict) -> np.ndarray:
        """Compute the whitened frequency domain strain array time shifted to geocenter.

        Args:
            parameters (dict): Dictionary containing sky position and time parameters.

        Returns:
            numpy.ndarray: Whitened frequency domain strain array time shifted at geocenter (detector, frequency).
        """
        time_delay_array = self.compute_time_delay_array(parameters)
        output = compute_time_shifted_frequency_domain_strain_array(
            frequency_array=self.frequency_array,
            frequency_mask=self.frequency_mask,
            frequency_domain_strain_array=self.whitened_frequency_domain_strain_array,
            time_delay_array=time_delay_array,
        )
        self._cached_whitened_frequency_domain_strain_array_at_geocenter = output
        return output

    # def transform_to_wavelet_domain(self, frequency_domain_data: np.ndarray) -> np.ndarray:
    #     """Transform frequency domain data to wavelet domain.

    #     Args:
    #         frequency_domain_data (numpy.ndarray): Frequency domain data array (detector, frequency).

    #     Returns:
    #         numpy.ndarray: Wavelet domain data array (detector, time, frequency).
    #     """
    #     return np.array(
    #         [
    #             transform_wavelet_freq(
    #                 data=frequency_domain_data[i],
    #                 sampling_frequency=self.sampling_frequency,
    #                 frequency_resolution=self.wavelet_frequency_resolution,
    #                 nx=self.wavelet_nx,
    #             )
    #             for i in range(len(self.interferometers))
    #         ]
    #     )

    # def get_wavelet_domain_strain_at_geocenter(self, parameters: dict) -> np.ndarray:
    #     """Compute the wavelet domain strain array at geocenter.

    #     Args:
    #         parameters (dict): Dictionary containing sky position and time parameters.

    #     Returns:
    #         numpy.ndarray: Wavelet domain strain array at geocenter (detector, time, frequency).
    #     """
    #     # Get whitened strain at geocenter (this will cache it)
    #     whitened_strain_geocenter = self.compute_whitened_strain_at_geocenter(parameters)

    #     # Transform to wavelet domain
    #     return self.transform_to_wavelet_domain(whitened_strain_geocenter)

    # def apply_whitening_to_matrix(self, matrix: np.ndarray) -> np.ndarray:
    #     """Apply whitening to a matrix using the stored PSDs.

    #     This is used for whitening antenna pattern matrices and other detector-based matrices.

    #     Args:
    #         matrix (numpy.ndarray): Matrix to whiten (detector, ...).

    #     Returns:
    #         numpy.ndarray: Whitened matrix.
    #     """
    #     # This will be used by AntennaPatternProcessor for whitening antenna patterns
    #     # For now, we'll implement basic whitening - this may need refinement
    #     whitened_matrix = np.zeros_like(matrix)
    #     for i, ifo in enumerate(self.interferometers):
    #         psd = self.power_spectral_density_array[i]
    #         # Apply whitening weights (simplified - may need more sophisticated approach)
    #         whitened_matrix[i] = matrix[i] / np.sqrt(psd[self.frequency_mask])
    #     return whitened_matrix

    def _validate_interferometers(self, interferometers: bilby.gw.detector.networks.InterferometerList):
        """Validate interferometers.

        Args:
            interferometers: A list of interferometers.

        Raises:
            ValueError: If interferometers do not have the same frequency resolution.
        """
        if not all(
            interferometer.frequency_array[1] - interferometer.frequency_array[0]
            == interferometers[0].frequency_array[1] - interferometers[0].frequency_array[0]
            for interferometer in interferometers[1:]
        ):
            raise ValueError("All interferometers must have the same delta_f.")

    def _validate_time_frequency_filter(self):
        """Validate the time frequency filter."""
        if self.time_frequency_filter is not None:
            ntime, nfreq = self.time_frequency_filter.shape
            assert (
                nfreq == self.tf_Nf
            ), f"The length of frequency axis = {nfreq} in the wavelet domain does not match the time frequency filter = {self.tf_Nf}."
            assert (
                ntime == self.tf_Nt
            ), f"The length of time axis = {ntime} in the wavelet domain does not match the time frequency filter = {self.tf_Nt}."
