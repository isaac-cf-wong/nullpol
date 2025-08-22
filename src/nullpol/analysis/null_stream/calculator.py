"""Null stream calculator for energy computations."""

from __future__ import annotations

import numpy as np

from .projections import (
    compute_gw_projector_masked,
    compute_null_projector_from_gw_projector,
    compute_projection_squared,
)


class NullStreamCalculator:
    """Modern null stream calculation using direct projection approach.

    This class implements the mathematically clean projection approach:
    1. Compute GW projector: P_gw = F(F†F)^(-1)F†
    2. Compute null projector: P_null = I - P_gw
    3. Apply null projection: E_null = d† P_null d

    This is a pure computational class with no component dependencies, using
    dependency injection for all data inputs.
    """

    def __init__(self):
        """Initialize the null stream calculator.

        No dependencies - this is a pure computational class that receives
        all required data via method parameters (dependency injection pattern).
        """
        pass

    # =========================================================================
    # PRIMARY INTERFACE METHODS
    # =========================================================================

    def compute_null_energy(
        self,
        whitened_antenna_pattern_matrix,
        whitened_time_frequency_strain_data,
        frequency_mask,
        time_frequency_filter,
    ):
        """Compute null energy using direct projection approach.

        This method implements the historical projection-based approach:
        1. Compute GW projector from antenna patterns
        2. Compute null projector as orthogonal complement
        3. Apply null projection to compute energy directly

        Args:
            whitened_antenna_pattern_matrix (numpy.ndarray): Whitened antenna patterns
                with shape (n_frequencies, n_detectors, n_modes).
            whitened_time_frequency_strain_data (numpy.ndarray): Whitened strain data
                with shape (n_detectors, n_time, n_frequencies).
            frequency_mask (numpy.ndarray): Boolean mask with shape (n_frequencies,).
            time_frequency_filter (numpy.ndarray): Boolean filter with shape (n_time, n_frequencies).

        Returns:
            float: Null energy computed via direct projection.
        """
        # Step 1: Compute GW projector P_gw = F(F†F)^(-1)F†
        gw_projector = self.compute_gw_projector(whitened_antenna_pattern_matrix, frequency_mask)

        # Step 2: Compute null projector P_null = I - P_gw
        null_projector = self.compute_null_projector(gw_projector)

        # Step 3: Apply null projection and compute squared magnitude
        null_energy_array = self.compute_projection_energy(
            whitened_time_frequency_strain_data, null_projector, time_frequency_filter
        )

        # Sum over all time-frequency pixels to get total null energy
        return np.sum(null_energy_array)

    # =========================================================================
    # COMPONENT METHODS (BUILDING BLOCKS)
    # =========================================================================

    def compute_gw_projector(self, whitened_antenna_pattern_matrix, frequency_mask):
        """Compute gravitational wave projector matrix.

        Args:
            whitened_antenna_pattern_matrix (numpy.ndarray): Whitened antenna patterns
                with shape (n_frequencies, n_detectors, n_modes).
            frequency_mask (numpy.ndarray): Boolean frequency mask with shape (n_frequencies,).

        Returns:
            numpy.ndarray: GW projector matrix with shape (n_frequencies, n_detectors, n_detectors).
        """
        return compute_gw_projector_masked(whitened_antenna_pattern_matrix, frequency_mask)

    def compute_null_projector(self, gw_projector):
        """Compute null projector as orthogonal complement to GW projector.

        Args:
            gw_projector (numpy.ndarray): GW projector matrix with shape
                (n_frequencies, n_detectors, n_detectors).

        Returns:
            numpy.ndarray: Null projector matrix P_null = I - P_gw.
        """
        return compute_null_projector_from_gw_projector(gw_projector)

    def compute_projection_energy(self, time_frequency_strain_data, projector, time_frequency_filter):
        """Compute energy of projected strain data.

        Args:
            time_frequency_strain_data (numpy.ndarray): Strain data in time-frequency domain
                with shape (n_detectors, n_time, n_frequencies).
            projector (numpy.ndarray): Projection operator with shape
                (n_frequencies, n_detectors, n_detectors).
            time_frequency_filter (numpy.ndarray): Time-frequency filter with shape
                (n_time, n_frequencies).

        Returns:
            numpy.ndarray: Projection energies for each time-frequency pixel with shape
                (n_time, n_frequencies).
        """
        return compute_projection_squared(time_frequency_strain_data, projector, time_frequency_filter)
