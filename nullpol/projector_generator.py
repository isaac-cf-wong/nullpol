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


class ProjectorGenerator(object):
    """Null projector generator."""

    def __init__(self, parameters=None, waveform_arguments=None):

        self.parameters = parameters

        self.polarization_str = waveform_arguments['polarization']
        self.polarization = np.sum([polarization_dict[k] for k in self.polarization_str], axis=0, dtype=bool)

        self.basis_str = waveform_arguments['basis']
        self.basis = np.sum([polarization_dict[k] for k in self.basis_str], axis=0, dtype=bool)

        self.interferometers = waveform_arguments['interferometers']
        self.detect_names = [interferometer.name for interferometer in self.interferometers]
        self.frequency_array = self.interferometers[0].frequency_array
        self.psd_array = np.array([interferometer.power_spectral_density_array for interferometer in self.interferometers])

    def get_null_projector(self, parameters):
        """Null projector.

        Parameters
        ----------
        parameters : dict
            Dictionary of waveform parameters with keys 'ra', 'dec', 'psi', 'geocent_time'.

        Returns
        -------
        array_like
            Null projector with shape (n_interferometers, n_interferometers, n_freqs).

        """
        antenna_pattern_matrix = (antenna_pattern.get_antenna_pattern_matrix(self.detect_names, parameters['ra'], parameters['dec'], parameters['psi'], parameters['geocent_time'], self.polarization))
        whitened_antenna_pattern_matrix = antenna_pattern.whiten_antenna_pattern_matrix(antenna_pattern_matrix, self.frequency_array, self.psd_array)
        whitened_antenna_pattern_matrix_new_basis = antenna_pattern.change_basis(whitened_antenna_pattern_matrix, self.basis, parameters['amp_phase_factor'])

        return null_projector.get_null_projector(whitened_antenna_pattern_matrix_new_basis)
    