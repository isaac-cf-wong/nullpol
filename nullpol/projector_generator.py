from . import null_projector
from . import antenna_pattern
import numpy as np

polarization_dict = {
    'p': (1, 0, 0, 0, 0, 0),
    'c': (0, 1, 0, 0, 0, 0),
    'b': (0, 0, 1, 0, 0, 0),
    'l': (0, 0, 0, 1, 0, 0),
    'x': (0, 0, 0, 0, 1, 0),
    'y': (0, 0, 0, 0, 0, 1),
}


class projector_generator(object):
    """Null projector generator."""

    def __init__(self, parameters=None, waveform_arguments=None):

        self.polarization_str = waveform_arguments['polarization']
        self.parameters = parameters
        self.interferometers = waveform_arguments['interferometers']

        request_pol = [polarization_dict[k] for k in self.polarization_str]
        
        self.polarization = np.sum(request_pol, axis=0, dtype=bool)

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
    