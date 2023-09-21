from . import null_projector
from . import antenna_pattern
import numpy as np

class projector_generator(object):
    """Null projector generator."""

    def __init__(self, parameters=None, waveform_arguments=None):

        self.polarization_str = waveform_arguments['polarization']
        self.parameters = parameters
        self.interferometers = waveform_arguments['interferometers']

        self.polarization = np.full(6, False, dtype=bool)
        if 'p' in self.polarization_str:
            self.polarization[0] = True
        if 'c' in self.polarization_str:
            self.polarization[1] = True
        if 'b' in self.polarization_str:
            self.polarization[2] = True
        if 'l' in self.polarization_str:
            self.polarization[3] = True
        if 'x' in self.polarization_str:
            self.polarization[4] = True
        if 'y' in self.polarization_str:
            self.polarization[5] = True

    def get_null_projector(self, interferometers, parameters):
        """Null projector.

        Parameters
        ----------
        interferometers : array_like
            Array of bilby.gw.detector.interferometer.Interferometer objects with same frequency array.
        parameters : dict
            Dictionary of waveform parameters with keys 'ra', 'dec', 'psi', 'geocent_time'.

        Returns
        -------
        array_like
            Null projector with shape (n_interferometers, n_polarization, n_freqs).

        """
        detectors = [interferometer.name for interferometer in interferometers]
        antenna_pattern_matrix = (antenna_pattern.antenna_pattern_matrix(detectors, parameters['ra'], parameters['dec'], parameters['psi'], parameters['geocent_time'], self.polarization))

        return null_projector.get_null_projector(interferometers, antenna_pattern_matrix)
    