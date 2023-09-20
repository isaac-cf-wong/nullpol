from . import null_projector
from . import antenna_pattern
import numpy as np

class projector_generator():
    """Null projector generator."""

    def __init__(self):
        pass

    def get_null_projector(self, interferometers, right_ascension, declination, polarization_angle, gps_time, polarization, frequency_array, psd):
        """Null projector.

        Parameters
        ----------
        interferometers : array_like
            Array of bilby.gw.detector.interferometer.Interferometer objects.
        right_ascension : float
            Right ascension in radians.
        declination : float
            Declination in radians.
        polarization_angle : float
            Polarization angle in radians.
        gps_time : float
            GPS time.
        polarization : array_like
            An array of polarization modes.
        frequency_array : array_like
            Frequency array of PSD.
        psd : array_like
            PSD.

        Returns
        -------
        array_like
            Null projector with shape (n_interferometers, n_polarization, n_freqs).

        """
        detectors = [interferometer.name for interferometer in interferometers]
        antenna_pattern_matrix = (antenna_pattern.antenna_pattern_matrix(detectors, right_ascension, declination, polarization_angle, gps_time, polarization))

        return null_projector.get_null_projector(antenna_pattern_matrix, frequency_array, psd)
    