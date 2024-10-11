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

def import_function(path):
    module_path, func_name = path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    func = getattr(module, func_name)
    return func

def get_file_extension(file_path):
    return Path(file_path).suffix

def main():
    default_config_file_path = pkg_resources.resource_filename('nullpol.tools', 'default_config_create_injection.ini')
    example_signal_parameters_path = pkg_resources.resource_filename('nullpol.tools', 'example_signal_parameters_create_injection.json')
    parser = ArgParser(default_config_files=[default_config_file_path])
    parser.add('-c', '--config', is_config_file=True, help='Path to custom config file.')
    parser.add('-o', '--outdir', type=str, help='Path of output.')
    parser.add('--label', type=str, help='Label of the injection')
    parser.add('--detectors', type=str, nargs="+", help='Detector prefix.')
    parser.add('--minimum-frequency', type=float, help='Minimum frequency in Hz.')
    parser.add('--signal-parameters', type=str, help='Path to a JSON file with a list of signal parameters.', default=example_signal_parameters_path)
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
    # Set the minimum frequency
    for interferometer in interferometers:
        interferometer.minimum_frequency = args.minimum_frequency
    
    # Check the extension of a file.
    if get_file_extension(args.signal_parameters) != ".json":
        raise ValueError('--signal-parameters needs to be a .json file.')
    # Load the signal parameters
    with open(args.signal_parameters, 'r') as f:
        signal_parameters = json.load(f)

    # Load the waveform arguments
    waveform_arguments = json.loads(args.waveform_arguments)

    # Load the frequency domain source model
    frequency_domain_source_model = import_function(args.frequency_domain_source_model)

    # Load the parameter conversion function
    parameter_conversion = import_function(args.parameter_conversion)

    waveform_generator = WaveformGenerator(duration=args.duration,
                                           sampling_frequency=args.sampling_frequency,
                                           frequency_domain_source_model=frequency_domain_source_model,
                                           parameter_conversion=parameter_conversion,
                                           waveform_arguments=waveform_arguments)
    
    # Create noise from PSD
    interferometers.set_strain_data_from_power_spectral_densities(
        sampling_frequency=args.sampling_frequency,
        duration=args.duration,
        start_time=args.start_time,
    )

    # Inject signal
    for signal_parameter in signal_parameters:
        interferometers.inject_signal(
            parameters=signal_parameter,
            waveform_generator=waveform_generator
        )

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
            interferometer.strain_data.freqency_domain_strain *= calibration_error

            # Update the noise PSD
            new_psd = interferometer.power_spectral_density.get_power_spectral_density_array(interferometer.frequency_array) * np.abs(calibration_error)

            # Save the PSD to disk.
            np.savetxt(outdir/f'{interferometer.name}:{args.label}-{args.start_time}-{args.duration}-psd.dat',
                       np.array(
                           [interferometer.frequency_array,
                            new_psd]
                       ).T)

    # Write the strain data    
    for interferometer in interferometers:
        ts = interferometer.strain_data.to_pycbc_timeseries()
        write_frame(str(outdir/f'{interferometer.name}:{args.label}-{args.start_time}-{args.duration}.gwf'), f'{interferometer.name}:STRAIN', ts)