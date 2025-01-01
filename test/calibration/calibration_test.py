import unittest
import tempfile
import numpy as np
from bilby.gw.detector import InterferometerList
from bilby.gw.prior import CalibrationPriorDict, PriorDict
from bilby.gw.detector.calibration import (CubicSpline,
                                           read_calibration_file)
from bilby.gw.detector.psd import PowerSpectralDensity
import tables
import os
import matplotlib.pyplot as plt
from pycbc.noise import noise_from_psd
from nullpol.calibration import build_calibration_lookup
from nullpol.psd import simulate_psd_from_bilby_psd
from nullpol.utility import logger
from nullpol.time_frequency_transform import (transform_wavelet_freq,
                                              get_shape_of_wavelet_transform)
from nullpol.null_stream import compute_whitened_time_frequency_domain_strain_array                                              


class TestCalibration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dirname = self.temp_dir.name
        seed = 12
        np.random.seed(seed)
        self.srate = 4096
        self.minimum_frequency = 20
        self.maximum_frequency = self.srate / 2
        self.seglen = 8
        self.n_nodes = 10
        self.number_of_response_curves = 10
        self.wavelet_frequency_resolution = 4.
        self.wavelet_frequency_array = np.arange(int(self.maximum_frequency/self.wavelet_frequency_resolution)) * self.wavelet_frequency_resolution
        self.wavelet_Nt, self.wavelet_Nf = get_shape_of_wavelet_transform(self.seglen, self.srate, self.wavelet_frequency_resolution)
        self.simulate_psd_nsample = 10
        self.wavelet_nx = 4.            
        self.ndet = 3
        self.time_frequency_filter = np.full((self.wavelet_Nt, self.wavelet_Nf), True)
        self.time_frequency_filter[:,:int(np.ceil(self.minimum_frequency/self.wavelet_frequency_resolution))] = False
        self.interferometers = InterferometerList(['H1', 'L1', 'V1'])
        self.interferometers.set_strain_data_from_power_spectral_densities(sampling_frequency=self.srate,
                                                                           duration=self.seglen,)
        self.injected_psds = {ifo.name: ifo.power_spectral_density_array.copy() for ifo in self.interferometers}
        for ifo in self.interferometers:
            self.injected_psds[ifo.name][~ifo.frequency_mask] = 0.
        for ifo in self.interferometers:
            ifo.calibration_model = CubicSpline(prefix=f"recalib_{ifo.name}_",
                                                minimum_frequency=self.minimum_frequency,
                                                maximum_frequency=self.maximum_frequency,
                                                n_points=self.n_nodes)
        # Temporary files
        self.envelope_files = {ifo.name: os.path.join(self.temp_dirname, f"{ifo.name}_envelope_file.txt") for ifo in self.interferometers}
        self.calibration_lookup_files = {ifo.name: os.path.join(self.temp_dirname, f"{ifo.name}_calibration_lookup_table.h5") for ifo in self.interferometers}
        self.calibration_psd_lookup_files = {ifo.name: os.path.join(self.temp_dirname, f"{ifo.name}_calibration_psd_lookup_table.h5") for ifo in self.interferometers}
        # Simulate an envelope file
        for ifo in self.interferometers:
            self.simulate_envelope_file(self.envelope_files[ifo.name])
        self.priors = PriorDict()
        for ifo in self.interferometers:
            self.priors.update(CalibrationPriorDict.from_envelope_file(
                envelope_file=self.envelope_files[ifo.name],
                minimum_frequency=self.minimum_frequency,
                maximum_frequency=self.maximum_frequency,
                n_nodes=self.n_nodes,
                label=ifo.name,
                correction_type='data',
            ))
        print(self.priors)
        # Generate the reference PSDs
        self.simulate_reference_psd()               

    def simulate_reference_psd(self):
        logger.info('Generating reference PSDs')
        delta_f = 1. / self.seglen
        self.reference_psds = [simulate_psd_from_bilby_psd(psd=ifo.power_spectral_density,
                                                           seglen=self.seglen,
                                                           srate=self.srate,
                                                           wavelet_frequency_resolution=self.wavelet_frequency_resolution,
                                                           nsample=self.simulate_psd_nsample,
                                                           nx=self.wavelet_nx) for ifo in self.interferometers]

    def simulate_envelope_file(self, fname):
        logger.info('Simulating calibration envelope files')
        n_nodes = 100
        frequency_array = np.exp(np.linspace(np.log(self.minimum_frequency), np.log(self.maximum_frequency), n_nodes))
        amplitude_median = np.full_like(frequency_array, 1.)
        phase_median = np.full_like(frequency_array, 0.)
        amplitude_lsigma = np.full_like(frequency_array, 1.)
        amplitude_usigma = np.full_like(frequency_array, 1.)
        phase_lsigma = np.full_like(frequency_array, 0.)
        phase_usigma = np.full_like(frequency_array, 0.)
        # amplitude_median = np.random.randn(n_nodes) * 2. + 2.
        # phase_median = np.random.randn(n_nodes) * 2. + 0.6
        # amplitude_lsigma = amplitude_median - np.random.randn(n_nodes)**2
        # amplitude_usigma = amplitude_median + np.random.randn(n_nodes)**2
        # phase_lsigma = phase_median - np.random.randn(n_nodes)**2
        # phase_usigma = phase_median + np.random.randn(n_nodes)**2
        envelope_data = np.concatenate([[frequency_array],
                                        [amplitude_median],
                                        [phase_median],
                                        [amplitude_lsigma],
                                        [phase_lsigma],
                                        [amplitude_usigma],
                                        [phase_usigma]]).T
        np.savetxt(fname, envelope_data, comments='# Frequency    Median Mag     Phase (Rad)    -1 Sigma Mag   -1 Sigma Phase +1 Sigma Mag   +1 Sigma Phase')

    def test_build_calibration_lookup_when_there_are_no_lookup_files(self):
        build_calibration_lookup(interferometers=self.interferometers,
                                 wavelet_frequency_resolution=self.wavelet_frequency_resolution,
                                 simulate_psd_nsample=self.simulate_psd_nsample,
                                 wavelet_nx=self.wavelet_nx,
                                 lookup_files=self.calibration_lookup_files,
                                 psd_lookup_files=self.calibration_psd_lookup_files,
                                 priors=self.priors,
                                 number_of_response_curves=self.number_of_response_curves,
                                 starting_index=0)
        # Check whether the calibrated PSD matches the expectation
        time_frequency_domain_strain_array = np.zeros((self.ndet, self.wavelet_Nt, self.wavelet_Nf))
        psd_array = np.zeros((self.ndet, self.wavelet_Nf))
        ifos = InterferometerList(['H1', 'L1', 'V1'])
        for i in range(self.ndet):
            calibration_draws, parameter_draws = read_calibration_file(filename=self.calibration_lookup_files[self.interferometers[i].name],
                                                                       frequency_array=self.interferometers[i].frequency_array,
                                                                       number_of_response_curves=self.number_of_response_curves,
                                                                       starting_index=0,
                                                                       correction_type='data')
            # Recalibrate the strain
            calibrated_frequency_domain_strain = self.interferometers[i].frequency_domain_strain / calibration_draws[0]
            # Transform to TF domain
            time_frequency_domain_strain_array[i,:,:] = transform_wavelet_freq(calibrated_frequency_domain_strain, self.wavelet_Nf, self.wavelet_Nt, self.wavelet_nx)
            with tables.open_file(self.calibration_psd_lookup_files[self.interferometers[i].name], 'r') as f:
                psd_array[i,:] = f.root.psd.psd_draws[0]
                # Whiten
        whitened_time_frequency_domain_strain_array = compute_whitened_time_frequency_domain_strain_array(time_frequency_domain_strain_array=time_frequency_domain_strain_array,
                                                                                                          psd_array=psd_array,
                                                                                                          time_frequency_filter=self.time_frequency_filter,
                                                                                                          srate=self.srate)
        print(whitened_time_frequency_domain_strain_array)                                                                                                        
        print(np.var(whitened_time_frequency_domain_strain_array[:,:int(np.ceil(self.minimum_frequency/self.wavelet_frequency_resolution))]))
        

    def test_build_calibration_lookup_when_there_are_lookup_files(self):
        pass

if __name__ == '__main__':
    unittest.main()