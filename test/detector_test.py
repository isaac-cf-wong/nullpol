import unittest
from nullpol.null_stream.antenna_pattern import get_antenna_pattern, get_antenna_pattern_matrix, whiten_antenna_pattern_matrix
from bilby.gw.detector import InterferometerList


class TestAntennaPattern(unittest.TestCase):
    def test_get_antenna_pattern(self):
        ifos = InterferometerList(['H1', 'L1'])
        ra = 0
        dec = 0
        psi = 0
        gps_time = 0
        polarization = [True, True, False, False, False, False]
        antenna_pattern = get_antenna_pattern(ifos[0], ra, dec, psi, gps_time, polarization)
        self.assertEqual(antenna_pattern.shape, (sum(polarization),))
    
    def test_get_antenna_pattern_matrix(self):
        ifos = InterferometerList(['H1', 'L1'])
        ra = 0
        dec = 0
        psi = 0
        gps_time = 0
        polarization = [True, True, False, False, False, False]
        antenna_pattern_matrix = get_antenna_pattern_matrix(ifos, ra, dec, psi, gps_time, polarization)
        self.assertEqual(antenna_pattern_matrix.shape, (2, sum(polarization)))

if __name__ == '__main__':
    unittest.main()
