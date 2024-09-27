from argparse import ArgumentParser
import numpy as np
import pandas as pd
from scipy.stats import chi2
from astropy.coordinates import SkyCoord
from ligo.skymap.io.fits import read_sky_map
from ligo.skymap.postprocess.crossmatch import crossmatch

from nullpol.null_stream.null_stream import get_null_stream, get_null_energy
from nullpol.detector.antenna_pattern import get_antenna_pattern_matrix, whiten_antenna_pattern_matrix
from nullpol.null_stream.null_projector import get_null_projector


""" SEE BELOW FOR INSTRUCTIONS IN LAUNCHING IN THE TERMINAL """


def p_null_energy_method(ifos, minimum_frequency, ra_true, dec_true, psi, geocent_time):
    """
    Get p-value using the "null energy method" described in:
    Pang et al. (2020) "Generic searches for alternative
    gravitational wave polarizations with networks of
    interferometric detectors." (https://arxiv.org/abs/2003.07375).
    ================================================================
    Parameters:
    -----------
    ifos: 'bilby.gw.detector.networks.InterferometerList'
        List of interferometers.
    minimum_frequency: float
        Minimum frequency.
    ra_true: float in the range (0, 2pi)
        True value of right ascension in radians (known from EM counterpart).
    dec_true: float in the range (-pi/2, pi/2)
        True value of declination angle in radians (known from EM counterpart).
    psi: float in the range (0, pi)
        Polarization angle of signal in radians.
    geocent_time: float
        Geocent time of signal.
    ============================================================================
    Output:
    -------
    p_value: float in the range (0, 1)
        The p-value to the "tensor-only" hypothesis.
    """
    maximum_frequency = ifos[0].maximum_frequency
    frequency_array = ifos[0].frequency_array
    psds = np.array([ifo.power_spectral_density_array for ifo in ifos])

    F_matrix_true = get_antenna_pattern_matrix(
        interferometers=ifos,
        right_ascension=ra_true,
        declination=dec_true,
        polarization_angle=psi,
        gps_time=geocent_time,
        polarization=[True, True, False, False, False, False], # tensor-only hypothesis
    )
    
    F_matrix_true_whitened = whiten_antenna_pattern_matrix(
        antenna_pattern_matrix=F_matrix_true,
        frequency_array=frequency_array,
        psds=psds,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )
    
    null_projector_true = get_null_projector(F_matrix_true_whitened)
    
    null_stream_true = get_null_stream(
        interferometers=ifos,
        null_projector=null_projector_true,
        ra=ra_true,
        dec=dec_true,
        gps_time=geocent_time,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )
    
    null_energy_true = get_null_energy(null_stream_true)
    
    n_freqs = frequency_array[
        (frequency_array >= minimum_frequency) & (frequency_array <= maximum_frequency)
    ].shape[0]
    n_ifos = len(ifos)
    dof = 2 * n_freqs * (n_ifos - 2)
    
    p_value = chi2.sf(2 * null_energy_true, dof)
    
    return p_value



# Generating ".fits" file:
# https://lscsoft.docs.ligo.org/bilby/api/bilby.gw.result.CompactBinaryCoalescenceResult.html#bilby.gw.result.CompactBinaryCoalescenceResult.plot_skymap

def p_sky_map_method(true_sky_pos, path_to_fits):
    """
    Get p-value using the "sky map method" described in:
    Pang et al. (2020) "Generic searches for alternative
    gravitational wave polarizations with networks of
    interferometric detectors." (https://arxiv.org/abs/2003.07375).
    ================================================================
    Parameters:
    -----------
    true_sky_pos: dict or 'astropy.coordinates.sky_coordinate.SkyCoord'
        Contains the true sky position of the source (right ascension,
        declination) and -- optionally -- (in case it is a dictionary)
        the units in which these physical quantities are given.
    path_to_fits: string
        Path to the '.fits' file generated from the posterior samples.
    ====================================================================
    Output:
    -------
    p_value: float in the range (0, 1)
        The p-value to the "tensor-only" hypothesis. The true sky
        position will fall on the (1 - p_value) credible contour
        specified by the posteriors from parameter estimation.
    """
    if isinstance(true_sky_pos, dict):
        unit = true_sky_pos['unit'] if 'unit' in true_sky_pos.keys() else 'rad'
        # If not specified, we're assuming both angles are given in radians (see line above).
        true_sky_pos = SkyCoord(true_sky_pos['ra'], true_sky_pos['dec'], unit=unit)
    elif not isinstance(true_sky_pos, SkyCoord):
        print('The argument "true_sky_pos" must either be a dictionary or of the type "astropy.coordinates.sky_coordinate.SkyCoord".')
    
    skymap = read_sky_map(path_to_fits, moc=True)
    
    p_value = 1 - crossmatch(skymap, true_sky_pos).searched_prob
    
    return p_value



def get_p_combined(p_values):
    """
    Getting the combined p-value coming from previous analyses' p-values.
    ======================================================================
    Parameters:
    -----------
    p_values: array_like w/ shape (n_p_values)
        Array of p-values coming from previous analyses.
    =====================================================
    Output:
    -------
    p_combined: float in the range (0, 1)
        Combined p-value coming from previous analyses' p-values.
    """
    if isinstance(p_values, list):
        p_values = np.asarray(p_values)
    elif not isinstance(p_values, np.ndarray):
        raise TypeError('The argument "p_values" must either be a list or a numpy array.')
    N = p_values.shape[0]
    dof = 2 * N # Chi square distribution's degree of freedom.
    
    log_p_values = np.log(p_values)
    test_statistic = - 2 * np.sum(log_p_values)
    
    p_combined = chi2.sf(test_statistic, dof)

    return p_combined



def main():
    
    parser = ArgumentParser(description='Frequentist test for non-tensorial polarizations (using the null energy method).')

    parser.add_argument('-ifos', '--interferometers', type=str, help='Path to pickle file containing InterferometerList.')
    parser.add_argument('-minfreq', '--minimum_frequency', type=float, help='Minimum frequency in Hz.')
    parser.add_argument('-ra', '--ra_true', type=float, help='True value of right ascension in radians (known from EM counterpart).')
    parser.add_argument('-dec', '--dec_true', type=float, help='True value of declination angle in radians (known from EM counterpart).')
    parser.add_argument('-psi', '--psi', type=float, help='Polarization angle of signal in radians.')
    parser.add_argument('-time', '--geocent_time', type=float, help='Geocent time of signal.')

    args = parser.parse_args()

    path_to_ifos = args.interferometers
    minimum_frequency = args.minimum_frequency
    ra_true = args.ra_true
    dec_true = args.dec_true
    psi = args.psi
    geocent_time = args.geocent_time
    
    ifos = pd.read_pickle(path_to_ifos)
    
    p_value = p_null_energy_method(ifos, minimum_frequency, ra_true, dec_true, psi, geocent_time)
    
    print("\np-value:", p_value, "\n")
    