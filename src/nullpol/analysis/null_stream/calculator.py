"""Null stream calculator for energy computations."""

from __future__ import annotations

import numpy as np

from ..antenna_patterns import AntennaPatternProcessor
from ..data_context import TimeFrequencyDataContext
from ..tf_transforms import transform_wavelet_freq
from .projections import compute_gw_projector, compute_null_projector, compute_null_stream


# pylint: disable=too-few-public-methods
class NullStreamCalculator:
    """Modern null stream calculation using direct projection approach.

    This class manages data context and antenna pattern processing along with
    null stream computations.

    Args:
        interferometers (list): List of interferometers.
        wavelet_frequency_resolution (float): The frequency resolution of the wavelet transform.
        wavelet_nx (int): The number of points in the wavelet transform.
        polarization_modes (list): List of polarization modes.
        polarization_basis (list, optional): List of polarization basis.
        time_frequency_filter (np.ndarray, optional): The time-frequency filter.
    """

    def __init__(
        self,
        interferometers,
        wavelet_frequency_resolution,
        wavelet_nx,
        polarization_modes,
        polarization_basis=None,
        time_frequency_filter=None,
    ):
        """Initialize the null stream calculator with data context and antenna pattern processor.

        Args:
            interferometers (list): List of interferometers.
            wavelet_frequency_resolution (float): The frequency resolution of the wavelet transform.
            wavelet_nx (int): The number of points in the wavelet transform.
            polarization_modes (list): List of polarization modes.
            polarization_basis (list, optional): List of polarization basis.
            time_frequency_filter (np.ndarray, optional): The time-frequency filter.
        """
        # Create the data context component - handles all data management
        self.data_context = TimeFrequencyDataContext(
            interferometers=interferometers,
            wavelet_frequency_resolution=wavelet_frequency_resolution,
            wavelet_nx=wavelet_nx,
            time_frequency_filter=time_frequency_filter,
        )

        # Initialize antenna pattern processor for polarization computations
        self.antenna_pattern_processor = AntennaPatternProcessor(
            polarization_modes=polarization_modes,
            polarization_basis=polarization_basis,
            interferometers=interferometers,
        )

    # =========================================================================
    # INTERNAL HELPER METHODS
    # =========================================================================

    def _compute_filtered_null_stream(self, parameters):
        """Compute the filtered null stream in time-frequency domain.

        This internal method performs the common computation steps shared by both
        compute_null_energy and compute_principal_null_components methods.

        Args:
            parameters (dict): Dictionary of parameters containing sky location, polarization, etc.

        Returns:
            tuple: A tuple containing:
                - filtered_null_strain (np.ndarray): Filtered null stream in time-frequency domain.
                - whitened_antenna_pattern_matrix (np.ndarray): Whitened antenna pattern matrix.
        """
        # Step 1: Get whitened strain data at geocenter in FREQUENCY domain
        # Shape: (n_detectors, n_frequencies)
        whitened_frequency_strain_data = self.data_context.compute_whitened_strain_at_geocenter(parameters)

        # Step 2: Compute whitened antenna patterns in frequency domain
        # Shape: (n_frequencies, n_detectors, n_modes)
        whitened_antenna_pattern_matrix = self.antenna_pattern_processor.compute_whitened_antenna_pattern_matrix(
            interferometers=self.data_context.interferometers,
            power_spectral_density_array=self.data_context.power_spectral_density_array,
            frequency_mask=self.data_context.frequency_mask,
            parameters=parameters,
        )

        # Step 3: Compute the GW signal projector for each frequency bin (masked)
        # Make sure gw_projector and whitened_frequency_strain_data have the same data type
        gw_projector = compute_gw_projector(whitened_antenna_pattern_matrix, self.data_context.frequency_mask).astype(
            whitened_frequency_strain_data.dtype
        )

        # Step 4: Compute the null projector (orthogonal complement to GW projector)
        null_projector = compute_null_projector(gw_projector)

        # Step 5: Project the whitened frequency-domain strain onto the null space
        null_stream_freq = compute_null_stream(
            whitened_frequency_strain_data, null_projector, self.data_context.frequency_mask
        )

        # Step 6: Transform the null stream to the time-frequency domain
        null_stream_time_freq = np.array(
            [
                transform_wavelet_freq(
                    data=null_stream_freq[i],
                    sampling_frequency=self.data_context.sampling_frequency,
                    frequency_resolution=self.data_context.wavelet_frequency_resolution,
                    nx=self.data_context.wavelet_nx,
                )
                for i in range(len(null_stream_freq))
            ]
        )

        # Step 7: Apply the time-frequency filter to the null stream
        filtered_null_strain = null_stream_time_freq * self.data_context.time_frequency_filter

        # Step 8: Recalculating the antenna patterns using the frequency mask of the null stream (in time-frequency domain)
        # Shape: (n_tf_freq_bins, n_detectors, n_modes)
        whitened_antenna_pattern_matrix = self.antenna_pattern_processor.compute_whitened_antenna_pattern_matrix(
            interferometers=self.data_context.interferometers,
            power_spectral_density_array=self.data_context.power_spectral_density_array,
            frequency_mask=self.data_context.time_frequency_filter[0, :],
            parameters=parameters,
        )

        return filtered_null_strain, whitened_antenna_pattern_matrix

    # =========================================================================
    # PRIMARY INTERFACE METHODS
    # =========================================================================

    def compute_null_energy(self, parameters):
        """Compute the total null energy from parameters.

        This method computes the whitened antenna patterns and strain data from the parameters,
        projects the whitened frequency-domain strain data onto the null space
        (orthogonal to the GW signal subspace), transforms to the time-frequency domain,
        applies a filter, and sums the squared magnitude to obtain the total null energy.

        Args:
            parameters (dict): Dictionary of parameters containing sky location, polarization, etc.

        Returns:
            float: The total null energy after projection and filtering.
        """
        # Compute the filtered null stream (Steps 1-7)
        filtered_null_strain, _ = self._compute_filtered_null_stream(parameters)

        # Step 8: Sum the squared magnitude to obtain the total null energy
        null_energy = np.sum(np.abs(filtered_null_strain) ** 2)

        return null_energy

    def compute_principal_null_components(self, parameters):
        """Compute the principal null components from parameters.

        This method computes the whitened antenna patterns and strain data from the parameters,
        projects the whitened frequency-domain strain data onto the null space,
        transforms to the time-frequency domain, applies a filter, and computes the principal
        null components using SVD.

        Args:
            parameters (dict): Dictionary of parameters containing sky location, polarization, etc.

        Returns:
            np.ndarray: The principal null components.
        """
        # Validate assumptions for this computation
        n_basis_modes = np.sum(self.antenna_pattern_processor.polarization_basis)
        assert n_basis_modes == 2, (
            f"Principal null component computation assumes exactly 2 polarization basis modes, "
            f"but got {n_basis_modes}. The current implementation uses [2:] slicing which is only "
            f"valid for 2 basis modes (typically 'plus' and 'cross')."
        )
        n_modes = np.sum(self.antenna_pattern_processor.polarization_modes)
        assert n_modes == n_basis_modes, (
            f"Principal null component computation assumes all polarization modes are basis modes, "
            f"but got {n_modes} modes and {n_basis_modes} basis modes. "
            f"Derived polarization modes are not supported in the current implementation."
        )

        # Compute the filtered null stream (Steps 1-7)
        filtered_null_strain, whitened_antenna_pattern_matrix = self._compute_filtered_null_stream(parameters)

        # Step 8: Compute the principal null components
        U, _S, _Vh = np.linalg.svd(whitened_antenna_pattern_matrix)

        return np.einsum("fdm, dtf -> mtf", np.conj(U), filtered_null_strain)[2, :, :]
