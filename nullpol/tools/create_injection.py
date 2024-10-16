from configargparse import ArgParser
import pkg_resources
import importlib
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
from pycbc.frame import write_frame
from bilby.gw.waveform_generator import WaveformGenerator
from bilby.gw.detector import InterferometerList
from bilby.gw.detector import PowerSpectralDensity
import re
from ..utility import logger

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
    default_config_file_path = pkg_resources.resource_filename('nullpol.tools', 'default_config_create_injection.ini')
    parser = ArgParser(default_config_files=[default_config_file_path])
    parser.add('-c', '--config', is_config_file=True, help='Path to custom config file.')
    parser.add('-o', '--outdir', type=str, help='Path of output.')
    parser.add('--label', type=str, help='Label of the injection')
    parser.add('--detectors', type=str, nargs="+", help='Detector prefix.')
    parser.add('--psds', type=json_loads_with_none, help='A dictionary of PSD files.')
    parser.add('--minimum-frequency', type=float, help='Minimum frequency in Hz.')
    parser.add('--signal-files', type=json_loads_with_none, help='A dictionary of files that contains the signal.')
    parser.add('--signal-file-channels', type=json_loads_with_none, help='A dictionary of channel names for the signal files.')
    parser.add('--signal-parameters', type=str, help='Path to a JSON file with a list of signal parameters.')
    parser.add('--waveform-arguments', type=str, help='A dictionary of additional arguments for the waveform model.')
    parser.add('--frequency-domain-source-model', type=str, help='Path to the frequency domain source function.')
    parser.add('--parameter-conversion', type=str, help="Parameter conversion function.")
    parser.add('--duration', type=int, help='Duration of the data in second.')
    parser.add('--start-time', type=int, help='GPS start time in second.')
    parser.add('--sampling-frequency', type=float, help='Sampling frequency in Hz.')
    parser.add('--calibration-errors', type=json.loads, help='A dictionary of calibration errors.')
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

        waveform_generator = WaveformGenerator(duration=args.duration,
                                            sampling_frequency=args.sampling_frequency,
                                            frequency_domain_source_model=frequency_domain_source_model,
                                            parameter_conversion=parameter_conversion,
                                            waveform_arguments=waveform_arguments)
    else:
        logger.info('frequency-domain-source-model is not provided. Not injecting signals from source model.')
        waveform_generator = None

    # Create noise from PSD
    interferometers.set_strain_data_from_power_spectral_densities(
        sampling_frequency=args.sampling_frequency,
        duration=args.duration,
        start_time=args.start_time,
    )

    # Inject signal
    if signal_parameters is not None and waveform_generator is not None:
        for i in range(len(signal_parameters)):
            interferometers.inject_signal(
                parameters=signal_parameters[i],
                waveform_generator=waveform_generator
            )
            logger.info(f'Signal {i+1}/{len(signal_parameters)} - Injected a signal with parameters: {signal_parameters[i]}')

    # Load the signal files if they are not empty
    if args.signal_files is not None:
        signal_interferometers = InterferometerList(args.detectors)
        for i in range(ndetector):
            signal_interferometer = signal_interferometers[i]
            interferometer = interferometers[i]
            if interferometer.name in args.signal_files and args.signal_files[interferometer.name] is not None:
                signal_file = args.signal_files[interferometer.name]
                signal_file_channel = args.signal_file_channels[interferometer.name]
                # Load frame file
                signal_interferometer.set_strain_data_from_frame_file(frame_file=signal_file,
                                                                      sampling_frequency=args.sampling_frequency,
                                                                      duration=args.duration,
                                                                      start_time=args.start_time,
                                                                      channel=signal_file_channel)
                # Add the signal to the interferometer
                interferometer.strain_data.frequency_domain_strain += signal_interferometer.strain_data.frequency_domain_strain
                logger.info(f'{interferometer.name} - Injected signal file {signal_file}.')

    outdir = Path(args.outdir)

    if args.calibration_errors is not None:
        # Apply the calibration errors to the frequency domain strain data.
        for interferometer in interferometers:
            # Load the calibration error
            calibration_error_data = np.loadtxt(args.calibration_errors[interferometer.name])

            # Interpolate the calibration errors
            calibration_error_interp = interp1d(calibration_error_data[:,0], calibration_error_data[:,1], kind='cubic', bounds_error=False, fill_value=1.)

            # Compute the calibration errors using the strain frequency array
            calibration_error = calibration_error_interp(interferometer.frequency_array)

            # Multiply the error to the frequency domain strain.
            interferometer.strain_data.freqency_domain_strain /= calibration_error

            # Update the noise PSD
            new_psd = interferometer.power_spectral_density.get_power_spectral_density_array(interferometer.frequency_array) / np.abs(calibration_error)**2

            # Save the PSD to disk.
            np.savetxt(outdir/f'{interferometer.name}-{args.label}-{args.start_time}-{args.duration}-psd.dat',
                       np.array(
                           [interferometer.frequency_array,
                            new_psd]
                       ).T)

    # Write the strain data
    for interferometer in interferometers:
        ts = interferometer.strain_data.to_pycbc_timeseries()
        output_path = str(outdir/f'{interferometer.name}-{args.label}-{args.start_time}-{args.duration}.gwf')
        channel_name = f'{interferometer.name}:STRAIN'
        write_frame(output_path, channel_name, ts)
        logger.info(f'{interferometer.name} strain data is written to the channel: {channel_name} in the file located at {output_path}.')
