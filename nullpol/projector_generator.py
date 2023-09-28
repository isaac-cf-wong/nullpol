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
        self.polarization_input = waveform_arguments['polarization']
        self.polarization = np.sum([polarization_dict[k] for k in self.polarization_input], axis=0, dtype=bool)

        if 'basis' in waveform_arguments: # if basis is not specified, use all polarization modes as basis
            self.basis_input = waveform_arguments['basis']
        else:
            self.basis_input = waveform_arguments['polarization']
        self.basis = np.sum([polarization_dict[k] for k in self.basis_input], axis=0, dtype=bool)
        self.basis_str = np.array(list(polarization_dict.keys()))[self.basis]
        self.additional_polarization_str = np.array(list(polarization_dict.keys()))[self.polarization][~self.basis[self.polarization]]

        self.interferometers = waveform_arguments['interferometers']
        self.detector_names = [interferometer.name for interferometer in self.interferometers]
        self.frequency_array = self.interferometers[0].frequency_array
        self.psd_array = np.array([interferometer.power_spectral_density_array for interferometer in self.interferometers])

    def get_amp_phase_factor_matrix(self, parameters):
        """
        Get amplitude and phase factor matrix.

        Parameters
        ----------
        parameters : dict
            Dictionary of waveform parameters

        Returns
        -------
        array_like
            Amplitude and phase factor matrix with shape (n_polarization-n_basis, n_basis, 2).
        """
        amp_phase_factor = np.zeros((self.polarization.sum()-self.basis.sum(), self.basis.sum(), 2))

        for i, additional_polarization in enumerate(self.additional_polarization_str):
            for j, basis in enumerate(self.basis_str):
                amp_phase_factor[i, j, 0] = parameters['amp_' + basis + additional_polarization]
                amp_phase_factor[i, j, 1] = parameters['phase_' + basis + additional_polarization]

        return amp_phase_factor

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
        antenna_pattern_matrix = (antenna_pattern.get_antenna_pattern_matrix(self.detector_names, parameters['ra'], parameters['dec'], parameters['psi'], parameters['geocent_time'], self.polarization))
        whitened_antenna_pattern_matrix = antenna_pattern.whiten_antenna_pattern_matrix(antenna_pattern_matrix, self.frequency_array, self.psd_array)
        self.amp_phase_factor = self.get_amp_phase_factor_matrix(parameters)
        whitened_antenna_pattern_matrix_new_basis = antenna_pattern.change_basis(whitened_antenna_pattern_matrix, self.basis, self.amp_phase_factor)

        return null_projector.get_null_projector(whitened_antenna_pattern_matrix_new_basis)
