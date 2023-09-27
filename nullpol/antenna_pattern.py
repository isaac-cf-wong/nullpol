from pycbc.detector import Detector
from lal import GreenwichMeanSiderealTime, ComputeDetAMResponseExtraModes
import numpy as np

def get_antenna_pattern(detector, right_ascension, declination, polarization_angle, gps_time, polarization):
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
        Antenna pattern for the given sky location and time with shape (n_polarization).
    """
    gmst = GreenwichMeanSiderealTime(gps_time)

    f_all = ComputeDetAMResponseExtraModes(Detector(detector, gps_time).response, right_ascension, declination, polarization_angle, gmst)
    
    return np.array(f_all)[polarization]

def get_antenna_pattern_matrix(detectors, right_ascension, declination, polarization_angle, gps_time, polarization):
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
        Antenna pattern matrix for the given sky location and time with shape (n_interferometers, n_polarization).
    """
    return np.array([get_antenna_pattern(detector, right_ascension, declination, polarization_angle, gps_time, polarization) for detector in detectors])

def whiten_antenna_pattern_matrix(antenna_pattern_matrix, frequency_array, psds):
    """
    Whiten antenna pattern matrix.

    Parameters
    ----------
    antenna_pattern_matrix : array_like
        Antenna pattern matrix with shape (n_interferometers, n_polarization).
    frequency_array : array_like
        Frequency array with shape (n_freqs).
    psds : array_like
        Power spectral density array with shape (n_interferometers, n_freqs).

    Returns
    -------
    whitened_antenna_pattern_matrix : array_like
        Whitened antenna pattern matrix with shape (n_interferometers, n_polarization, n_freqs).
    """
    df = frequency_array[1] - frequency_array[0]

    whitening_factor = 1/np.sqrt(psds/(2*df)) # shape (n_interferometers, n_freqs)

    return np.einsum('ij, ik -> ijk', antenna_pattern_matrix, whitening_factor) # shape (n_interferometers, n_polarization, n_freqs)

def change_basis(whitened_antenna_pattern_matrix, basis, amp_phase_factor):
    """
    Change basis of whitened antenna pattern matrix.

    Parameters
    ----------
    whitened_antenna_pattern_matrix : array_like
        Whitened antenna pattern matrix with shape (n_interferometers, n_polarization, n_freqs).
    basis : array_like
        Array of booleans for basis modes.
    amp_phase_factor : array_like
        Array of amplitude and phase factors for basis modes with shape (n_polarization-n_basis, n_basis, 2).

    Returns
    -------
    whitened_antenna_pattern_matrix_new_basis : array_like
        Whitened antenna pattern matrix with shape (n_interferometers, n_basis, n_freqs).
    """
    multiplicative_factor = amp_phase_factor[:, :, 0] + 1j * amp_phase_factor[:, :, 1] # shape (n_polarization-n_basis, n_basis)
    additional_terms = np.einsum('ijk, jl -> ilk', whitened_antenna_pattern_matrix[:, ~basis, :], multiplicative_factor) # shape (n_interferometers, n_basis, n_freqs)
    
    return whitened_antenna_pattern_matrix[:, basis, :] + additional_terms # shape (n_interferometers, n_basis, n_freqs)
