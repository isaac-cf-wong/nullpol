from configargparse import ArgParser
import sys
import numpy as np
from pycbc.types.frequencyseries import FrequencySeries
from scipy.interpolate import interp1d
from ..psd import simulate_psd_from_psd
from ..utility import logger
from .. import __version__

def main():
    parser = ArgParser()
    parser.add('psd_file', type=str, nargs='?', help="PSD file.", default=None)
    parser.add('-o', '--output', type=str, help='Path of output.')
    parser.add('--sampling-freuency', type=float, help="Sampling frequency in Hz.")
    parser.add('--duration', type=float, help="Duration in second.")
    parser.add('--nsample', type=int, help="Number of samples.")
    parser.add('--wavelet-frequency-resolution', type=float, help="Wavelet frequency resolution in Hz.")
    parser.add('--wavelet-sharpness', type=float, help="Sharpness of wavelet.", default=4.)
    parser.add('--seed', type=int, help="Seed.")
    parser.add('-v', '--version', action='store_true')

    args = parser.parse_args()

    if args.version:
        logger.info(__version__)
        sys.exit(0)

    # Read the PSD file. Assume a text file.
    psd_data = np.loadtxt(args.psd_file)
    freq = psd_data[:,0]
    psd = psd_data[:,1]
    # Interpolate the PSD.
    tlen = int(args.sampling_frequency * args.duration)
    flen = tlen // 2 + 1
    delta_f = 1. / args.duration
    sample_frequencies = np.arange(flen) * delta_f
    psd_interp = interp1d(freq, psd, kind='linear', bounds_error=False, fill_value=0.)(sample_frequencies)
    psd_pycbc = FrequencySeries(psd_interp, delta_f=delta_f)
    if args.seed is not None:
        np.random.seed(args.seed)
    simulated_psd = simulate_psd_from_psd(psd_pycbc, args.duration, args.sampling_frequency, args.wavelet_frequency_resolution, args.nsample, nx=4.)
    simulated_freq = np.arange(len(simulated_psd)) * args.wavelet_frequency_resolution
    output_concat = np.concatenate(([simulated_freq], [simulated_psd]), axis=0).T
    np.savetxt(args.output, output_concat)