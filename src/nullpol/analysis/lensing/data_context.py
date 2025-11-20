"""Time-frequency data context for strong lensing analysis with two images."""

from __future__ import annotations

from bilby.gw.detector.networks import InterferometerList
import numpy as np

from ..data_context import TimeFrequencyDataContext

class LensingTimeFrequencyDataContext(TimeFrequencyDataContext):
    """Data context for strong lensing analysis with two images.

    Manages two sets of interferometers observing two lensed images of the same signal.
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
        self._interferometers_1 = InterferometerList(interferometers[0])
        self._interferometers_2 = InterferometerList(interferometers[1])
        super().__init__(
            interferometers=self._interferometers_1 + self._interferometers_2,
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

    def compute_time_delay_array(self, parameters: dict) -> np.ndarray:
        """Compute time delay array with lensing time delay applied to second set.

        Args:
            parameters (dict): Dictionary containing 'ra', 'dec', 'geocent_time', and 'time_delay'.

        Returns:
            np.ndarray: Array of time delays.
        """

        time_delay_array_1 = [
                ifo.time_delay_from_geocenter(
                    ra=parameters["ra"], dec=parameters["dec"], time=parameters["geocent_time"]
                )
                for ifo in self.interferometers_1
            ]
        time_delay_array_2 = [
                ifo.time_delay_from_geocenter(
                    ra=parameters["ra"], dec=parameters["dec"], time=parameters["geocent_time"]+parameters["time_delay"]
                )
                for ifo in self.interferometers_2
            ]
        return np.concatenate([time_delay_array_1, time_delay_array_2])
