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

    def __init__(self, parameters=None, waveform_arguments=None, **kwargs):
        self.polarization_input = waveform_arguments['polarization']
        if not self.polarization_input:
            raise ValueError('Polarization modes must be specified.')
        self.polarization = np.sum([polarization_dict[k] for k in self.polarization_input], axis=0, dtype=bool)

        if 'basis' in waveform_arguments:
            self.basis_input = waveform_arguments['basis']
            if not self.basis_input:
                raise ValueError('Basis polarization modes must be specified.')
            if not all([p in self.polarization_input for p in self.basis_input]):
                raise ValueError('Basis polarization modes must be in the polarization modes.')
        else:
            self.basis_input = self.polarization_input # if basis is not specified, use all polarization modes as basis
        self.basis = np.sum([polarization_dict[k] for k in self.basis_input], axis=0, dtype=bool)[self.polarization]
        self.basis_str = np.array(list(polarization_dict.keys()))[self.polarization][self.basis]
        self.additional_polarization_str = np.array(list(polarization_dict.keys()))[self.polarization][~self.basis]

        self.interferometers = waveform_arguments['interferometers']
        if len(self.interferometers) > np.sum(self.basis):
            raise ValueError('Number of interferometers must be less than or equal to the number of basis polarization modes.')
        delta_f = self.interferometers[0].frequency_array[1] - self.interferometers[0].frequency_array[0]
        if not all([interferometer.frequency_array[1] - interferometer.frequency_array[0] == delta_f for interferometer in self.interferometers[1:]]):
            raise ValueError('All interferometers must have the same delta_f.')
        self.frequency_array = self.interferometers[0].frequency_array
        self.psd_array = np.array([np.interp(self.frequency_array, interferometer.power_spectral_density.frequency_array, interferometer.power_spectral_density.psd_array) for interferometer in self.interferometers])
        self.minimum_frequency = waveform_arguments['minimum_frequency']
        if not all([interferometer.frequency_array[0] <= self.minimum_frequency for interferometer in self.interferometers]):
            raise ValueError('minimum_frequency must be greater than or equal to the minimum frequency of all interferometers.')
        self.maximum_frequency = waveform_arguments['maximum_frequency']
        if not all([interferometer.frequency_array[-1] >= self.maximum_frequency for interferometer in self.interferometers]):
            raise ValueError('maximum_frequency must be less than or equal to the maximum frequency of all interferometers.')
        # check if maximum_frequency is less than the Nyquist frequency
        if not all([interferometer.sampling_frequency >= 2*self.maximum_frequency for interferometer in self.interferometers]):
            raise ValueError('maximum_frequency must be less than or equal to the Nyquist frequency of all interferometers.')

    def _get_amp_phase_factor_matrix(self, parameters):
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

    def null_projector(self, parameters):
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
        antenna_pattern_matrix = antenna_pattern.get_antenna_pattern_matrix(self.interferometers, parameters['ra'], parameters['dec'], parameters['psi'], parameters['geocent_time'], self.polarization)
        whitened_antenna_pattern_matrix = antenna_pattern.whiten_antenna_pattern_matrix(antenna_pattern_matrix, self.frequency_array, self.psd_array, self.minimum_frequency, self.maximum_frequency)
        self.amp_phase_factor = self._get_amp_phase_factor_matrix(parameters)
        whitened_antenna_pattern_matrix_new_basis = antenna_pattern.change_basis(whitened_antenna_pattern_matrix, self.basis, self.amp_phase_factor)

        return null_projector.get_null_projector(whitened_antenna_pattern_matrix_new_basis)
