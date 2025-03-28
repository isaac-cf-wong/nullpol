from __future__ import annotations

import unittest

import numpy as np

from nullpol.null_stream import compute_gw_projector_masked


class TestProjector(unittest.TestCase):
    def setUp(self):
        seed = 12
        np.random.seed(seed)

    def test_compute_gw_projector_masked(self):
        whitened_antenna_pattern_matrix = np.random.randn(128, 3, 2) + 1.j*np.random.randn(128, 3, 2)
        frequency_mask = np.full(128, True)
        frequency_mask[:20] = False
        output = compute_gw_projector_masked(whitened_antenna_pattern_matrix=whitened_antenna_pattern_matrix,
                                             frequency_mask=frequency_mask)
        expected_output = np.zeros((128, 3, 3), dtype=whitened_antenna_pattern_matrix.dtype)
        F = whitened_antenna_pattern_matrix[20:]
        F_dagger = np.conj(np.transpose(F, [0,2,1]))
        FdF =  F_dagger @ F
        expected_output[20:] = F @ np.linalg.inv(FdF) @ F_dagger
        self.assertTrue(np.allclose(output, expected_output))

if __name__ == '__main__':
    unittest.main()
