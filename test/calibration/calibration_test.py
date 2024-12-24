import unittest
import tempfile
import numpy as np
from bilby.gw.detector import InterferometerList
from bilby.gw.prior import CalibrationPriorDict, PriorDict
from bilby.gw.detector.calibration import CubicSpline
from nullpol.calibration import build_calibration_lookup


class TestCalibration(unittest.TestCase):
    def setUp(self):
        seed = 12
        np.random.seed(seed)
        srate = 4096
        self.minimum_frequency = 20
        self.maximum_frequency = srate / 2
        seglen = 8
        n_nodes = 10
        
        frequency_array = np.arange(int(srate*seglen/2 + 1)) / seglen
        # Simulate an envelope file
        self.envelope_files = [tempfile.NamedTemporaryFile(suffix='.txt') for _ in range(3)]
        for i in range(3):
            self.simulate_envelope_file(self.envelope_files[i].name)
        self.interferometers = InterferometerList(['H1', 'L1', 'V1'])
        self.interferometers.set_strain_data_from_power_spectral_densities(sampling_frequency=srate,
                                                                           duration=seglen,)
        for ifo in self.interferometers:
            ifo.calibration_model = CubicSpline(prefix=f"recalib_{ifo.name}_",
                                                minimum_frequency=self.minimum_frequency,
                                                maximum_frequency=self.maximum_frequency,
                                                n_points=n_nodes)
        self.priors = PriorDict()
        for i in range(len(self.interferometers)):
            self.priors.update(CalibrationPriorDict.from_envelope_file(
                envelope_file=self.envelope_files[i].name,
                minimum_frequency=self.minimum_frequency,
                maximum_frequency=self.maximum_frequency,
                n_nodes=n_nodes,
                label=self.interferometers[i].name,
            ))        

    def simulate_envelope_file(self, fname):
        n_nodes = 20
        frequency_array = np.exp(np.linspace(np.log(self.minimum_frequency), np.log(self.maximum_frequency), n_nodes))
        amplitude_median = np.random.randn(n_nodes) * 0.1 + 1.
        phase_median = np.random.randn(n_nodes) * 0.1
        amplitude_lsigma = np.random.randn(n_nodes)**2 * 0.1
        amplitude_lsigma = amplitude_median - amplitude_lsigma
        amplitude_usigma = np.random.randn(n_nodes)**2 * 0.1
        amplitude_usigma = amplitude_median + amplitude_usigma
        phase_lsigma = np.random.randn(n_nodes)**2 * 0.1
        phase_lsigma = phase_median - phase_lsigma
        phase_usigma = np.random.randn(n_nodes)**2 * 0.1
        phase_usigma = phase_median - phase_usigma
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
                                 wavelet_frequency_resolution=16,
                                 simulate_psd_nsample=10,
                                 wavelet_nx=4.,
                                 lookup_files=None,
                                 psd_lookup_files=None,
                                 priors=self.priors,
                                 number_of_response_curves=10,
                                 starting_index=0)

    def test_build_calibration_lookup_when_there_are_lookup_files(self):
        pass

if __name__ == '__main__':
    unittest.main()