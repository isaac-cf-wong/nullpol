from . projector_generator import ProjectorGenerator
from . import antenna_pattern
from . import null_projector
import numpy as np

class LensingProjectorGenerator(ProjectorGenerator):
    """Null projector generator for strong lensing with 2 images."""

    def __init__(self, parameters=None, waveform_arguments=None):

        super().__init__(parameters, waveform_arguments)

    def null_projector(self, parameters, interferometers_1, interferometers_2, frequency_array, psd_array_1, psd_array_2, minimum_frequency, maximum_frequency):
        """Null projector for strong lensing with 2 images.

        Parameters
        ----------
        parameters : dict
            Dictionary of waveform parameters with keys 'ra', 'dec', 'psi', 'geocent_time', 'geocent_time_2', 'lensing_amp', 'lensing_phase'.
        interferometers_1 : list
            List of bilby.gw.detector.Interferometer for the first image.
        interferometers_2 : list
            List of bilby.gw.detector.Interferometer for the second image.
        frequency_array : array_like
            Frequency array with shape (n_freqs).
        psd_array_1 : array_like
            Power spectral density array for the interferometers of the first image with shape (n_interferometers, n_freqs).
        psd_array_2 : array_like
            Power spectral density array for the interferometers of the second image with shape (n_interferometers, n_freqs).
        minimum_frequency : float
            Minimum frequency.
        maximum_frequency : float
            Maximum frequency.

        Returns
        -------
        array_like
            Null projector with shape (n_interferometers, n_interferometers, n_freqs)

        """
        antenna_pattern_matrix_1 = antenna_pattern.get_antenna_pattern_matrix(interferometers_1, parameters['ra'], parameters['dec'], parameters['psi'], parameters['geocent_time_1'], self.polarization)
        whitened_antenna_pattern_matrix_1 = antenna_pattern.whiten_antenna_pattern_matrix(antenna_pattern_matrix_1, frequency_array, psd_array_1, minimum_frequency, maximum_frequency)
        
        antenna_pattern_matrix_2 = antenna_pattern.get_antenna_pattern_matrix(interferometers_2, parameters['ra'], parameters['dec'], parameters['psi'], parameters['geocent_time_2'], self.polarization)
        whitened_antenna_pattern_matrix_2 = antenna_pattern.whiten_antenna_pattern_matrix(antenna_pattern_matrix_2, frequency_array, psd_array_2, minimum_frequency, maximum_frequency)

        frequency_array = frequency_array[(frequency_array >= self.minimum_frequency) & (frequency_array <= self.maximum_frequency)]
        time_delay = parameters['geocent_time_2'] - interferometers_2.start_time - parameters['geocent_time_1'] + interferometers_1.start_time
        lensing_factor = parameters['amp_lensing'] * np.exp(-1j * np.pi * parameters['phase_lensing'] + 2 * np.pi * 1j * time_delay * frequency_array) # shape (n_freqs)
        
        whitened_antenna_pattern_matrix = np.concatenate((whitened_antenna_pattern_matrix_1, np.einsum('ijk, k -> ijk', whitened_antenna_pattern_matrix_2, lensing_factor)), axis=0) # shape (n_interferometers, n_polarization, n_freqs)

        self.amp_phase_factor = self._get_amp_phase_factor_matrix(parameters)
        whitened_antenna_pattern_matrix_new_basis = antenna_pattern.change_basis(whitened_antenna_pattern_matrix, self.basis, self.amp_phase_factor)

        return null_projector.get_null_projector(whitened_antenna_pattern_matrix_new_basis)
