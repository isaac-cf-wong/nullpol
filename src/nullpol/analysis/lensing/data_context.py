"""Time-frequency data context for strong lensing analysis with two images."""

from __future__ import annotations

import numpy as np
from bilby.gw.detector.networks import InterferometerList

from ..data_context import TimeFrequencyDataContext

NUMBER_OF_LENSED_IMAGES = 2


class _LensingInterferometerList(InterferometerList):
    """Interferometer list that permits different image-epoch start times."""

    def _check_interferometers(self) -> None:
        """Require identical frequency grids while allowing distinct epoch starts."""
        if not self:
            return

        interferometer_ref = self[0]
        for attribute in ("duration", "sampling_frequency"):
            values = [getattr(interferometer.strain_data, attribute) for interferometer in self]
            if not all(value == values[0] for value in values[1:]):
                raise ValueError(f"The {attribute} of all interferometers must be the same.")

        frequency_array_ref = interferometer_ref.frequency_array
        for interferometer in self[1:]:
            frequency_array = interferometer.frequency_array
            if frequency_array.shape != frequency_array_ref.shape or not np.array_equal(
                frequency_array, frequency_array_ref
            ):
                raise ValueError("The frequency arrays of all interferometers must be the same.")


class LensingTimeFrequencyDataContext(TimeFrequencyDataContext):
    """Data context for strong lensing analysis with two images.

    Manages two sets of interferometers observing two lensed images of the same signal.
    The image networks may have different segment start times, while the detectors
    within an image must share one exact start time. Both images must have matching
    frequency grids so their frequency-domain data can be combined. The context
    aligns image two to image one's FFT time reference as part of the existing
    detector-to-geocenter phase shifts.

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
        self._validate_unique_interferometers([*self._interferometers_1, *self._interferometers_2])
        self._validate_image_start_times(self._interferometers_1)
        self._validate_image_start_times(self._interferometers_2)
        self._inter_image_start_time_offset = (
            self._interferometers_2[0].strain_data.start_time - self._interferometers_1[0].strain_data.start_time
        )
        super().__init__(
            interferometers=[*self._interferometers_1, *self._interferometers_2],
            wavelet_frequency_resolution=wavelet_frequency_resolution,
            wavelet_nx=wavelet_nx,
            time_frequency_filter=time_frequency_filter,
        )

    def _create_interferometer_list(self, interferometers):
        """Create the two-epoch network without enforcing a shared start time."""
        return _LensingInterferometerList(interferometers)

    @staticmethod
    def _validate_image_start_times(interferometers: InterferometerList) -> None:
        """Require sample-aligned FFT origins within each image network."""
        try:
            start_times = np.asarray(
                [interferometer.strain_data.start_time for interferometer in interferometers],
                dtype=float,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("All interferometers within an image must have the same start_time.") from error
        if not np.all(np.isfinite(start_times)) or not np.all(start_times == start_times[0]):
            raise ValueError("All interferometers within an image must have the same start_time.")

    @staticmethod
    def _validate_unique_interferometers(interferometers: list) -> None:
        """Reject a detector-data object reused for more than one image row."""
        if len({id(interferometer) for interferometer in interferometers}) != len(interferometers):
            raise ValueError("Each lensed image must use distinct interferometer data objects.")

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
        return self._inter_image_start_time_offset

    def compute_time_delay_array(self, parameters: dict) -> np.ndarray:
        """Compute phase shifts to the common image-one geocentric frame.

        The image-two shifts include both their detector-to-geocenter delays
        and the relative delay between the two FFT segment origins. The base
        data context applies these shifts in its existing frequency-domain
        alignment operation, so a single time-frequency filter has one time
        reference for every detector row.

        Args:
            parameters (dict): Dictionary containing 'ra', 'dec', 'geocent_time', and 'delta_t'.

        Returns:
            np.ndarray: Array of frequency-domain phase-shift delays.
        """
        time_delay_array_1 = [
            ifo.time_delay_from_geocenter(ra=parameters["ra"], dec=parameters["dec"], time=parameters["geocent_time"])
            for ifo in self.interferometers_1
        ]
        relative_image_delay = parameters["delta_t"] - self.inter_image_start_time_offset
        time_delay_array_2 = [
            ifo.time_delay_from_geocenter(
                ra=parameters["ra"], dec=parameters["dec"], time=parameters["geocent_time"] + parameters["delta_t"]
            )
            + relative_image_delay
            for ifo in self.interferometers_2
        ]
        return np.concatenate([time_delay_array_1, time_delay_array_2])
