"""Antenna pattern processing for gravitational wave analysis.

This module provides the AntennaPatternProcessor class that manages antenna pattern
computations, polarization encoding, and calibration corrections.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from .base import (
    get_antenna_pattern_matrix,
    get_collapsed_antenna_pattern_matrix,
    relative_amplification_factor_helper,
    relative_amplification_factor_map,
)
from .conditioning import (
    compute_calibrated_whitened_antenna_pattern_matrix,
    compute_whitened_antenna_pattern_matrix_masked,
)
from .encoding import (
    encode_polarization,
)

from ...utils import NullpolError


class AntennaPatternProcessor:
    """Antenna pattern processing for time-frequency analysis.

    This class handles antenna pattern computations, polarization encoding, calibration
    corrections, and relative amplification factors. It provides a clean interface for
    antenna pattern operations without embedding data management logic.

    Args:
        polarization_modes (list): List of polarization modes.
        polarization_basis (list, optional): List of polarization basis.
        interferometers (list): List of interferometers for validation.
    """

    def __init__(
        self,
        polarization_modes,
        polarization_basis=None,
        interferometers=None,
    ):
        # Encode the polarization labels
        self._polarization_modes, self._polarization_basis, self._polarization_derived = encode_polarization(
            polarization_modes, polarization_basis
        )

        # Validate detector count vs polarization bases
        if interferometers is not None:
            self._validate_detector_count(interferometers)

        # Collapse the polarization encoding
        self._polarization_basis_collapsed = np.array(
            [self.polarization_basis[i] for i in range(len(self.polarization_modes)) if self.polarization_modes[i]]
        ).astype(bool)
        self._polarization_derived_collapsed = np.array(
            [self.polarization_derived[i] for i in range(len(self.polarization_modes)) if self.polarization_modes[i]]
        ).astype(bool)

        # Compute the relative amplification factor map
        self._relative_amplification_factor_map = relative_amplification_factor_map(
            self.polarization_basis, self.polarization_derived
        )

    def _validate_detector_count(self, interferometers):
        """Validate that the number of detectors exceeds the number of polarization bases.

        Args:
            interferometers: List of interferometers.

        Raises:
            NullpolError: If detector count is insufficient.
        """
        if len(interferometers) <= np.sum(self._polarization_basis):
            raise NullpolError(
                f"Number of detectors = {len(interferometers)} has to be greater than the number of polarization bases = {np.sum(self._polarization_basis)}."
            )

    @property
    def polarization_modes(self):
        """Polarization modes.

        Returns:
            numpy array: A boolean array that encodes the polarization modes.
        """
        return self._polarization_modes

    @property
    def polarization_basis(self):
        """Polarization basis.

        Returns:
            numpy array: A boolean array that encodes the polarization basis.
        """
        return self._polarization_basis

    @property
    def polarization_derived(self):
        """Derived polarization modes.

        Returns:
            numpy array: A boolean array that encodes the derived polarization modes.
        """
        return self._polarization_derived

    @property
    def polarization_basis_collapsed(self):
        """A collapsed boolean array of the polarization basis.
        The modes not in polarization_modes are removed.

        Returns:
            numpy array: A collapsed boolean array of the polarization basis.
        """
        return self._polarization_basis_collapsed

    @property
    def polarization_derived_collapsed(self):
        """A collapsed boolean array of the derived polarization modes.
        The modes not in polarization_modes are removed.

        Returns:
            numpy array: A collapsed boolean array of the derived polarization modes.
        """
        return self._polarization_derived_collapsed

    @property
    def relative_amplification_factor_map(self):
        """A map to the relative amplification factor.

        Returns:
            numpy array: A map to the relative amplification factor (detector, mode).
        """
        return self._relative_amplification_factor_map

    def compute_antenna_pattern_matrix(self, interferometers, parameters: Dict[str, Any]):
        """Compute the antenna pattern matrix.

        Args:
            interferometers: List of interferometers.
            parameters: Dictionary of parameters containing 'ra', 'dec', 'psi', 'geocent_time'.

        Returns:
            numpy array: Antenna pattern matrix.
        """
        # Evaluate the antenna pattern function
        F_matrix = get_antenna_pattern_matrix(
            interferometers,
            right_ascension=parameters["ra"],
            declination=parameters["dec"],
            polarization_angle=parameters["psi"],
            gps_time=parameters["geocent_time"],
            polarization=self.polarization_modes,
        )

        # Evaluate the collapsed antenna pattern function
        # Compute the relative amplification factor
        if self.relative_amplification_factor_map.size > 0:
            relative_amplification_factor = relative_amplification_factor_helper(
                self.relative_amplification_factor_map, parameters
            )
            F_matrix = get_collapsed_antenna_pattern_matrix(
                F_matrix,
                self.polarization_basis_collapsed,
                self.polarization_derived_collapsed,
                relative_amplification_factor,
            )
        return F_matrix

    def compute_calibration_factor_matrix(
        self,
        interferometers,
        frequency_domain_strain_array,
        masked_frequency_array,
        frequency_mask,
        parameters: Dict[str, Any],
    ):
        """Compute the calibration factor matrix.

        Args:
            interferometers: List of interferometers.
            frequency_domain_strain_array: Frequency domain strain data.
            masked_frequency_array: Masked frequency array.
            frequency_mask: Boolean frequency mask.
            parameters: Dictionary of parameters.

        Returns:
            numpy array: Calibration factor array.
        """
        output = np.zeros_like(frequency_domain_strain_array)

        for i, interferometer in enumerate(interferometers):
            calibration_errors = interferometer.calibration_model.get_calibration_factor(
                frequency_array=masked_frequency_array,
                prefix=f"recalib_{interferometer.name}_",
                **parameters,
            )
            output[i, frequency_mask] = calibration_errors
        return output

    def compute_whitened_antenna_pattern_matrix(
        self, interferometers, power_spectral_density_array, frequency_mask, parameters: Dict[str, Any]
    ):
        """Compute the whitened antenna pattern matrix.

        Args:
            interferometers: List of interferometers.
            power_spectral_density_array: Power spectral density array.
            frequency_mask: Boolean frequency mask.
            parameters: Dictionary of parameters.

        Returns:
            numpy array: Whitened antenna pattern matrix.
        """
        # Compute the antenna pattern matrix
        antenna_pattern_matrix = self.compute_antenna_pattern_matrix(interferometers, parameters)

        # Compute the whitened antenna pattern matrix
        return compute_whitened_antenna_pattern_matrix_masked(
            antenna_pattern_matrix=antenna_pattern_matrix,
            psd_array=power_spectral_density_array,
            frequency_mask=frequency_mask,
        )

    def compute_calibrated_whitened_antenna_pattern_matrix(
        self,
        interferometers,
        power_spectral_density_array,
        frequency_domain_strain_array,
        masked_frequency_array,
        frequency_mask,
        parameters: Dict[str, Any],
    ):
        """Compute the calibrated whitened antenna pattern matrix.

        Args:
            interferometers: List of interferometers.
            power_spectral_density_array: Power spectral density array.
            frequency_domain_strain_array: Frequency domain strain data.
            masked_frequency_array: Masked frequency array.
            frequency_mask: Boolean frequency mask.
            parameters: Dictionary of parameters.

        Returns:
            numpy array: Calibrated whitened antenna pattern matrix.
        """
        # Compute the whitened antenna pattern matrix
        whitened_antenna_patten_matrix = self.compute_whitened_antenna_pattern_matrix(
            interferometers, power_spectral_density_array, frequency_mask, parameters
        )

        # Compute the calibration factor
        calibration_factor = self.compute_calibration_factor_matrix(
            interferometers, frequency_domain_strain_array, masked_frequency_array, frequency_mask, parameters
        )

        # Apply calibration corrections
        return compute_calibrated_whitened_antenna_pattern_matrix(
            frequency_mask=frequency_mask,
            whitened_antenna_pattern_matrix=whitened_antenna_patten_matrix,
            calibration_error_matrix=calibration_factor,
        )
