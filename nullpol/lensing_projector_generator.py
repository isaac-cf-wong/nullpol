from . projector_generator import ProjectorGenerator
from . import antenna_pattern
from . import null_projector
import numpy as np

class StrongLensingProjectorGenerator(ProjectorGenerator):
    """Null projector generator for strong lensing with 2 images."""

    def __init__(self, parameters=None, waveform_arguments=None):
        super().__init__(parameters, waveform_arguments)

        self.interferometers_second_image = waveform_arguments['interferometers_second_image']
        self.psd_array_second_image = np.array([np.interp(self.frequency_array, interferometer.power_spectral_density.frequency_array, interferometer.power_spectral_density.psd_array) for interferometer in self.interferometers_second_image])

    def null_projector(self, parameters):
        """Null projector for strong lensing with 2 images.

        Parameters
        ----------
        parameters : dict
            Dictionary of waveform parameters with keys 'ra', 'dec', 'psi', 'geocent_time', 'geocent_time_2', 'lensing_amp', 'lensing_phase'.

        Returns
        -------
        array_like
            Null projector with shape (n_interferometers * 2, n_interferometers * 2, n_freqs)

        """
        antenna_pattern_matrix_first = antenna_pattern.get_antenna_pattern_matrix(self.interferometers, parameters['ra'], parameters['dec'], parameters['psi'], parameters['geocent_time'], self.polarization)
        whitened_antenna_pattern_matrix_first = antenna_pattern.whiten_antenna_pattern_matrix(antenna_pattern_matrix_first, self.frequency_array, self.psd_array, self.minimum_frequency, self.maximum_frequency)
        
        antenna_pattern_matrix_second = antenna_pattern.get_antenna_pattern_matrix(self.interferometers, parameters['ra'], parameters['dec'], parameters['psi'], parameters['geocent_time_2'], self.polarization)
        whitened_antenna_pattern_matrix_second = antenna_pattern.whiten_antenna_pattern_matrix(antenna_pattern_matrix_second, self.frequency_array, self.psd_array_second_image, self.minimum_frequency, self.maximum_frequency)

        frequency_array = self.frequency_array[(self.frequency_array >= self.minimum_frequency) & (self.frequency_array <= self.maximum_frequency)]
        time_delay = parameters['geocent_time_2'] - parameters['geocent_time']
        lensing_factor = parameters['amp_lensing'] * np.exp(1j *parameters['phase_lensing'] + 2 * np.pi * 1j * time_delay * frequency_array) # shape (n_freqs)
        
        whitened_antenna_pattern_matrix = np.concatenate((whitened_antenna_pattern_matrix_first, np.einsum('ijk, k -> ijk', whitened_antenna_pattern_matrix_second, lensing_factor)), axis=0) # shape (n_interferometers * 2, n_polarization, n_freqs)

        self.amp_phase_factor = self._get_amp_phase_factor_matrix(parameters)
        whitened_antenna_pattern_matrix_new_basis = antenna_pattern.change_basis(whitened_antenna_pattern_matrix, self.basis, self.amp_phase_factor)

        return null_projector.get_null_projector(whitened_antenna_pattern_matrix_new_basis)
