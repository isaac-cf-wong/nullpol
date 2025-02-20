from bilby.gw.waveform_generator import WaveformGenerator
from bilby.gw.source import lal_binary_black_hole
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters


DEFAULT_BBH_WAVEFORM_ARGUMENTS = {"waveform_approximant": "IMRPhenomPv2",
                                  "reference_frequency": 50}


def create_injection(interferometers,
                     duration,
                     sampling_frequency,
                     start_time,
                     parameters=None,
                     noise_type='zero_noise',
                     frequency_domain_source_model=lal_binary_black_hole,
                     waveform_arguments=DEFAULT_BBH_WAVEFORM_ARGUMENTS):
    """A helper function to inject a mock signal into interferometers.

    Parameters
    ----------
    interferometers: InterferometerList
        A list of interferometers.
    duration: float
        Duration of data in second.
    sampling_frequency: float
        Sampling frequency in Hz.
    start_time: float
        Start time of the data segemtn in second.
    parameters: dict
        A dictionary of injection parameters.        
    noise_type: str
        Type of noise. Supported options: ['zero_noise', 'gaussian', 'noise']
    frequency_domain_source_model: callable
        A function that returns the frequency domain polarizations.
    waveform_arguments: dict
        A dictionary of additional waveform arguments.
    """
    if noise_type == 'gaussian':
        interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time)
    elif noise_type == 'zero_noise':
        interferometers.set_strain_data_from_zero_noise(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time)
    elif noise_type == 'noise':
        interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time)
        return
    else:
        raise ValueError(f'noise_type={noise_type} is not supported.')
    # Create a waveform generator
    waveform_generator = WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        start_time=start_time,
        frequency_domain_source_model=frequency_domain_source_model,
        parameters=None,
        parameter_conversion=convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments)
    interferometers.inject_signal(parameters=parameters,
                                  waveform_generator=waveform_generator)
