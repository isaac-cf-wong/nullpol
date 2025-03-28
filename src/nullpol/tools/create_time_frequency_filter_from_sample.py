from configargparse import ArgParser
import pkg_resources
import importlib
import json
from pathlib import Path
import numpy as np
import healpy as hp
from bilby.gw.waveform_generator import WaveformGenerator
from bilby.gw.detector import InterferometerList
from bilby.gw.detector import PowerSpectralDensity
from ..utils import logger
from ..time_frequency_transform import (transform_wavelet_freq,
                                        transform_wavelet_freq_quadrature)


def import_function(path):
    module_path, func_name = path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    func = getattr(module, func_name)
    return func


def get_file_extension(file_path):
    return Path(file_path).suffix


def json_loads_with_none(value):
    # Replace 'None' with 'null' to make it valid JSON
    value = value.replace('None', 'null')
    return json.loads(value)


def main():
    default_config_file_path = pkg_resources.resource_filename('nullpol.tools', 'default_config_create_time_frequency_filter_from_sample.ini')
    parser = ArgParser(default_config_files=[default_config_file_path])
    parser.add('-c', '--config', is_config_file=True, help='Path to custom config file.')
    parser.add('-o', '--output', type=str, help='Path of output.')
    parser.add('--detectors', type=str, nargs="+", help='Detector prefix.')
    parser.add('--psds', type=json_loads_with_none, help='A dictionary of PSD files.')
    parser.add('--minimum-frequency', type=float, help='Minimum frequency in Hz.')
    parser.add('--signal-parameters', type=str, help='Path to a JSON file with a list of signal parameters.')
    parser.add('--waveform-arguments', type=str, help='A dictionary of additional arguments for the waveform model.')
    parser.add('--frequency-domain-source-model', type=str, help='Path to the frequency domain source function.')
    parser.add('--parameter-conversion', type=str, help="Parameter conversion function.")
    parser.add('--duration', type=int, help='Duration of the data in second.')
    parser.add('--start-time', type=int, help='GPS start time in second.')
    parser.add('--sampling-frequency', type=float, help='Sampling frequency in Hz.')
    parser.add('--nside', type=int, help='nside should be a power of 2. 12 * nside * nside sky pixels are generated.')
    parser.add('--nx', type=float, help='Sharpness of wavelet.', default=4.)
    parser.add('--wavelet-df', type=float, help='Frequency resolution of wavelet transform in Hz.', default=16)
    parser.add('--threshold', type=float, help='Threshild to apply the filter.', default=0.1)
    parser.add('--generate-config', help='Generate default config file and exit.', is_write_out_config_file_arg=True)

    args = parser.parse_args()

    # Create the interferometer
    interferometers = InterferometerList(args.detectors)
    ndetector = len(interferometers)
    # Set the minimum frequency
    for interferometer in interferometers:
        interferometer.minimum_frequency = args.minimum_frequency

    # Set the PSDs if provided.
    if args.psds is not None:
        for i in range(ndetector):
            detector_name = interferometers[i].name
            if detector_name in args.psds and args.psds[detector_name] is not None:
                # Update the PSD.
                interferometers[i].power_spectral_density = PowerSpectralDensity.from_power_spectral_density_file(args.psds[detector_name])
                logger.info(f'{detector_name} PSD file loaded: {args.psds[detector_name]}.')
            else:
                logger.info(f'{detector_name} PSD is not provided. Default ASD file: {interferometers[i].power_spectral_density.asd_file} or default PSD file: {interferometers[i].power_spectral_density.psd_file} is used.')

    # Load the signal parameters
    if args.signal_parameters is not None:
        # Check the extension of a file.
        if get_file_extension(args.signal_parameters) != ".json":
            raise ValueError('--signal-parameters needs to be a .json file.')
        with open(args.signal_parameters, 'r') as f:
            signal_parameters = json.load(f)
    else:
        signal_parameters = None     

    # Construct waveform generator
    if args.frequency_domain_source_model is not None:
        # Load the frequency domain source model
        frequency_domain_source_model = import_function(args.frequency_domain_source_model)

        # Load the waveform arguments
        if args.waveform_arguments is not None:
            waveform_arguments = json.loads(args.waveform_arguments)
        else:
            waveform_arguments = None

        if args.parameter_conversion is not None:
            # Load the parameter conversion function
            parameter_conversion = import_function(args.parameter_conversion)
        else:
            parameter_conversion = None

        waveform_generator = WaveformGenerator(
            duration=args.duration,
            sampling_frequency=args.sampling_frequency,
            frequency_domain_source_model=frequency_domain_source_model,
            parameter_conversion=parameter_conversion,
            waveform_arguments=waveform_arguments)
    else:
        logger.info('frequency-domain-source-model is not provided. Not injecting signals from source model.')
        waveform_generator = None

    # Set zero noise
    interferometers.set_strain_data_from_zero_noise(
        sampling_frequency=args.sampling_frequency,
        duration=args.duration,
        start_time=args.start_time)

    # Inject signal
    if signal_parameters is not None and waveform_generator is not None:
        for i in range(len(signal_parameters)):
            interferometers.inject_signal(
                parameters=signal_parameters[i],
                waveform_generator=waveform_generator
            )
            logger.info(f'Signal {i+1}/{len(signal_parameters)} - Injected a signal with parameters: {signal_parameters[i]}')

    # Construct the whitened frequency-domain strain.
    whitened_frequency_domain_strains = []
    for interferometer in interferometers:
        frequency_domain_strain = interferometer.frequency_domain_strain
        power_spectral_density_array = interferometer.power_spectral_density_array
        scaling_factor = args.duration / 2
        whitened_frequency_domain_strains.append(np.divide(
            interferometer.frequency_domain_strain,
            np.sqrt(power_spectral_density_array * scaling_factor),
            out=np.zeros_like(frequency_domain_strain),
            where=power_spectral_density_array != 0.))

    # Construct the sky-maximized time-frequency filter
    wavelet_Nf = int(args.sampling_frequency / 2 / args.wavelet_df)
    wavelet_Nt = int(args.sampling_frequency * args.duration / wavelet_Nf)
    time_frequency_map = np.zeros((wavelet_Nt, wavelet_Nf))
    npix = hp.nside2npix(args.nside)
    middle_time = args.start_time + args.duration / 2
    maximized_time_frequency_map = np.zeros((wavelet_Nt, wavelet_Nf))
    for ipix in range(npix):
        theta, phi = hp.pix2ang(args.nside, ipix)
        ra = phi
        dec = np.pi / 2 - theta
        time_frequency_map = np.zeros((wavelet_Nt, wavelet_Nf))
        for i in range(ndetector):
            # Copy the frequency domain strain data.
            whitened_frequency_domain_strain_copy = whitened_frequency_domain_strains[i].copy()
            time_shift = interferometers[i].time_delay_from_geocenter(ra, dec, middle_time)
            frequency_mask = interferometers[i].frequency_mask
            frequencies = interferometers[i].frequency_array[frequency_mask]
            whitened_frequency_domain_strain_copy[frequency_mask] *= np.exp(1j * 2 * np.pi * time_shift * frequencies)
            # Transform to time-frequency domain
            whitened_wavelet_domain_strain = transform_wavelet_freq(
                data=whitened_frequency_domain_strain_copy,
                sampling_frequency=args.sampling_frequency,
                frequency_resolution=args.wavelet_df,
                nx=args.nx)
            whitened_wavelet_domain_strain_quadrature = transform_wavelet_freq_quadrature(
                data=whitened_frequency_domain_strain_copy,
                sampling_frequency=args.sampling_frequency,
                frequency_resolution=args.wavelet_df,
                nx=args.nx)
            whitened_wavelet_domain_power = whitened_wavelet_domain_strain ** 2 + whitened_wavelet_domain_strain_quadrature ** 2
            time_frequency_map += whitened_wavelet_domain_power
        maximized_time_frequency_map = np.maximum(maximized_time_frequency_map, time_frequency_map)
        logger.info(f'Maximizing signal power over the sky sphere - {ipix+1}/{npix}.')

    # Perform the clustering on the sky-maximized time-frequency map.
    # Apply a threshold
    time_frequency_filter = maximized_time_frequency_map >= args.threshold
    # Remove the frequency content beyond the range
    # Always remove the Nyquist frequency
    time_frequency_filter[:, -1] = 0.
    # Remove the components below the minimum frequency.
    freq_low_idx = int(np.ceil(args.minimum_frequency / args.wavelet_df))
    time_frequency_filter[:, :freq_low_idx] = 0.
    # Save the file to disk.
    np.save(args.output, time_frequency_filter)
    logger.info(f'Time-frequency filter is written to {args.output}.')
