"""Time-frequency data context for strong lensing analysis with two images."""

from __future__ import annotations

import numpy as np
from bilby.gw.detector.networks import InterferometerList

from ..data_context import TimeFrequencyDataContext

NUMBER_OF_LENSED_IMAGES = 2


class _LensingInterferometerList(InterferometerList):
    """Interferometer list that permits different start times for separate images."""

    def _check_interferometers(self) -> None:
        """Require matching frequency grids while allowing distinct epoch starts."""
        for attribute in ("duration", "sampling_frequency"):
            values = [getattr(interferometer.strain_data, attribute) for interferometer in self]
            if values and not all(np.isclose(value, values[0]) for value in values[1:]):
                raise ValueError(f"The {attribute} of all interferometers must be the same.")


class LensingTimeFrequencyDataContext(TimeFrequencyDataContext):
    """Data context for strong lensing analysis with two images.

    Manages two sets of interferometers observing two lensed images of the same signal.
    The image networks may have different segment start times, but must have matching
    duration and sampling frequency so their frequency-domain data can be combined.
    Handles differential time delays between the two images.

    Args:
        interferometers (list): List of two sublists of interferometers.
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
        """Initialize the two-image data context."""
        if not (
            isinstance(interferometers, list)
            and len(interferometers) == NUMBER_OF_LENSED_IMAGES
            and all(
                isinstance(image_interferometers, list) and image_interferometers
                for image_interferometers in interferometers
            )
        ):
            raise ValueError("interferometers must be a list of two lists of non-empty interferometers")

        self._interferometers_1 = InterferometerList(interferometers[0])
        self._interferometers_2 = InterferometerList(interferometers[1])
        super().__init__(
            interferometers=_LensingInterferometerList([*self._interferometers_1, *self._interferometers_2]),
            wavelet_frequency_resolution=wavelet_frequency_resolution,
            wavelet_nx=wavelet_nx,
            time_frequency_filter=time_frequency_filter,
        )

    @property
    def interferometers_1(self) -> InterferometerList:
        """First set of interferometers.

        Returns:
            InterferometerList: First set of interferometers.
        """
        return self._interferometers_1

    @property
    def interferometers_2(self) -> InterferometerList:
        """Second set of interferometers.

        Returns:
            InterferometerList: Second set of interferometers.
        """
        return self._interferometers_2

    @property
    def inter_image_start_time_offset(self) -> float:
        """Difference between the second and first image segment start times."""
        return self.interferometers_2[0].strain_data.start_time - self.interferometers_1[0].strain_data.start_time

    def compute_time_delay_array(self, parameters: dict) -> np.ndarray:
        """Compute detector-to-geocenter delays at each image's arrival time.

        The inter-image phase delay is applied by LensingNullStreamCalculator,
        accounting for the difference between the two segment start times.

        Args:
            parameters (dict): Dictionary containing 'ra', 'dec', 'geocent_time', and 'delta_t'.

        Returns:
            np.ndarray: Array of time delays.
        """
        time_delay_array_1 = [
            ifo.time_delay_from_geocenter(ra=parameters["ra"], dec=parameters["dec"], time=parameters["geocent_time"])
            for ifo in self.interferometers_1
        ]
        time_delay_array_2 = [
            ifo.time_delay_from_geocenter(
                ra=parameters["ra"], dec=parameters["dec"], time=parameters["geocent_time"] + parameters["delta_t"]
            )
            for ifo in self.interferometers_2
        ]
        return np.concatenate([time_delay_array_1, time_delay_array_2])
