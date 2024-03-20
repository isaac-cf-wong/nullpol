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

    def null_projector(self, parameters, interferometers, frequency_array, psd_array, minimum_frequency, maximum_frequency):
        """Null projector.

        Parameters
        ----------
        parameters : dict
            Dictionary of waveform parameters with keys 'ra', 'dec', 'psi', 'geocent_time'.
        interferometers : list
            List of bilby.gw.detector.Interferometer.
        frequency_array : array_like
            Frequency array with shape (n_freqs).
        psd_array : array_like
            Power spectral density with shape (n_interferometers, n_freqs).
        minimum_frequency : float
            Minimum frequency.
        maximum_frequency : float
            Maximum frequency.

        Returns
        -------
        array_like
            Null projector with shape (n_interferometers, n_interferometers, n_freqs).

        """
        antenna_pattern_matrix = antenna_pattern.get_antenna_pattern_matrix(interferometers, parameters['ra'], parameters['dec'], parameters['psi'], parameters['geocent_time'], self.polarization)
        whitened_antenna_pattern_matrix = antenna_pattern.whiten_antenna_pattern_matrix(antenna_pattern_matrix, frequency_array, psd_array, minimum_frequency, maximum_frequency)
        
        self.amp_phase_factor = self._get_amp_phase_factor_matrix(parameters)
        whitened_antenna_pattern_matrix_new_basis = antenna_pattern.change_basis(whitened_antenna_pattern_matrix, self.basis, self.amp_phase_factor)

        calibration = np.array([interferometer.calibration_model.get_calibration_factor(frequency_array, prefix='recalib_{}_'.format(self.name), **parameters) for interferometer in interferometers])
        whitened_antenna_pattern_matrix_new_basis_calibrated = np.einsum('ijk, ik -> ijk', whitened_antenna_pattern_matrix_new_basis, calibration)

        return null_projector.get_null_projector(whitened_antenna_pattern_matrix_new_basis_calibrated)
