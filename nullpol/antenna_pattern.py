from pycbc.detector import Detector

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
        An array of polarization modes.

    Returns
    -------
    antenna_pattern : array_like
        Antenna pattern for the given sky location and time.
    """
    det = Detector(detector)
    
    det.antenna_pattern(right_ascension, declination, polarization_angle, gps_time, )
    
    return None

def antenna_pattern_matrix(detectors, right_ascension, declination, polarization_angle, gps_time):
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

    Returns
    -------
    antenna_pattern_matrix : array_like
        Antenna pattern matrix for the given sky location and time.
    """
    return None

