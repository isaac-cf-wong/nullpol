from pycbc.detector import Detector
from lal import GreenwichMeanSiderealTime, ComputeDetAMResponseExtraModes
import numpy as np

def antenna_pattern(detector, right_ascension, declination, polarization_angle, gps_time, polarization):
    """
    Get antenna pattern for a given sky location and time.

    Parameters
    ----------
    detector : str
        Detector name.
    right_ascension : float
        Right ascension in radians.
    declination : float
        Declination in radians.
    polarization_angle : float
        Polarization angle in radians.
    gps_time : float
        GPS time.
    polarization : array_like
        Array of booleans for polarization modes.

    Returns
    -------
    antenna_pattern : array_like
        Antenna pattern for the given sky location and time.
    """
    gmst = GreenwichMeanSiderealTime(gps_time)

    f_all = ComputeDetAMResponseExtraModes(Detector(detector, gmst).response, right_ascension, declination, polarization_angle, gmst)
    
    return np.array(f_all)[polarization]

def antenna_pattern_matrix(detectors, right_ascension, declination, polarization_angle, gps_time, polarization):
    """
    Get antenna pattern matrix for a given sky location and time.

    Parameters
    ----------
    detectors : list
        List of detector names.
    right_ascension : float
        Right ascension in radians.
    declination : float
        Declination in radians.
    polarization_angle : float
        Polarization angle in radians.
    gps_time : float
        GPS time.
    polarization : array_like
        Array of booleans for polarization modes.

    Returns
    -------
    antenna_pattern_matrix : array_like
        Antenna pattern matrix for the given sky location and time.
    """
    antenna_pattern_matrix = []
    for detector in detectors:
        antenna_pattern_matrix.append(antenna_pattern(detector, right_ascension, declination, polarization_angle, gps_time, polarization))
    antenna_pattern_matrix = np.array(antenna_pattern_matrix)
    
    return antenna_pattern_matrix
