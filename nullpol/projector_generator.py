from . import null_projector
from . import antenna_pattern
import numpy as np

class projector_generator(object):
    """Null projector generator."""

    def __init__(self, parameters=None, waveform_arguments=None):
        pass

    def get_null_projector(self, interferometers, parameters, polarization):
        """Null projector.

        Parameters
        ----------
        interferometers : array_like
            Array of bilby.gw.detector.interferometer.Interferometer objects with same frequency array.
        parameters : dict
            Dictionary of waveform parameters with keys 'ra', 'dec', 'psi', 'geocent_time'.
        polarization : array_like
            An array of polarization modes.

        Returns
        -------
        array_like
            Null projector with shape (n_interferometers, n_polarization, n_freqs).

        """
        detectors = [interferometer.name for interferometer in interferometers]
        antenna_pattern_matrix = (antenna_pattern.antenna_pattern_matrix(detectors, parameters['ra'], parameters['dec'], parameters['psi'], parameters['geocent_time'], polarization))

        return null_projector.get_null_projector(interferometers, antenna_pattern_matrix)
    