import unittest
from nullpol.null_stream.projector_generator import ProjectorGenerator

class TestNullProjector(unittest.TestCase):
    def test_init(self):
        waveform_arguments = {'polarization': ['p', 'c', 'b', 'l', 'x', 'y'], 'basis': ['p', 'c']}
        projector_generator = ProjectorGenerator(waveform_arguments=waveform_arguments)
        self.assertListEqual(projector_generator.polarization_input, ['p', 'c', 'b', 'l', 'x', 'y'])
        self.assertListEqual(projector_generator.basis_input, ['p', 'c'])
        self.assertListEqual(projector_generator.basis_str.tolist(), ['p', 'c'])
        self.assertListEqual(projector_generator.additional_polarization_str.tolist(), ['b', 'l', 'x', 'y'])
        self.assertListEqual(projector_generator.polarization.tolist(), [True, True, True, True, True, True])
        self.assertListEqual(projector_generator.basis.tolist(), [True, True, False, False, False, False])

    def test_get_amp_phase_factor_matrix(self):
        waveform_arguments = {'polarization': ['p', 'c', 'b', 'l', 'x', 'y'], 'basis': ['p', 'c']}
        projector_generator = ProjectorGenerator(waveform_arguments=waveform_arguments)
        parameters = {'amp_pb': 1, 'phase_pb': 0, 'amp_pl': 1, 'phase_pl': 0, 'amp_px': 1, 'phase_px': 0, 'amp_py': 1, 'phase_py': 0, 'amp_cb': 1, 'phase_cb': 0, 'amp_cl': 1, 'phase_cl': 0, 'amp_cx': 1, 'phase_cx': 0, 'amp_cy': 1, 'phase_cy': 0}
        amp_phase_factor_matrix = projector_generator._get_amp_phase_factor_matrix(parameters)
        self.assertListEqual(amp_phase_factor_matrix.tolist(), [[[1, 0], [1, 0]], [[1, 0], [1, 0]], [[1, 0], [1, 0]], [[1, 0], [1, 0]]])

if __name__ == '__main__':
    unittest.main()
