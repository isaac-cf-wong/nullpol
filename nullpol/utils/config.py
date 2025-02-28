import configparser
from typing import Union, Any
from bilby_pipe.utils import (convert_string_to_dict,
                              convert_string_to_tuple,
                              convert_string_to_list)
from .convert_type import (convert_string_to_bool,
                           convert_string_to_float,
                           convert_string_to_int)


class ConfigParser:
    """A parser to read configuration file.

    Args:
        str: Path to a configuration file.
    """
    def __init__(self, ini: str):
        with open(ini, 'r') as f:
            file_content = "[root]\n" + f.read()
        self._config_parser = configparser.RawConfigParser()
        self._config_parser.read_string(file_content)

    def __getitem__(self, key: str) -> Union[Any, None]:
        """Get value.

        Args:
            key (str): Key

        Returns:
            Any: The item or None if the key does not exist.
        """
        return self._config_parser['root'].get(key, None)

    def __setitem__(self, key: str, item: str):
        """Set value.

        Args:
            key (str): Key.
            item (str): Item.
        """
        self._config_parser['root'][key] = str(item)

    @property
    def calibration_model(self) -> Union[str, None]:
        """Get the calibration model.

        Returns:
            str: Calibration model.
        """
        return self['calibration-model']

    @calibration_model.setter
    def calibration_model(self, value: str):
        """Set the calibration model.

        Args:
            value (str): Calibration model.
        """
        self['calibration-model'] = value

    @property
    def spline_calibration_envelope_dict(self) -> Union[dict, None]:
        """Get the spline calibration envelope dictionary.

        Returns
            dict: Spline calibration envelope dictionary.
        """
        return convert_string_to_dict(self['spline-calibration-envelope-dict'])

    @spline_calibration_envelope_dict.setter
    def spline_calibration_envelope_dict(self, value: dict):
        """Set the spline calibration envelope dictionary.

        Args:
            value (dict): Spline calibration envelope dictionary.
        """
        self['spline_calibration_envelope_dict'] = value

    @property
    def spline_calibration_nodes(self) -> Union[int, None]:
        """Get the number of spline calibration nodes.

        Returns:
            int: Number of spline calibration nodes.
        """
        return convert_string_to_int(self['spline-calibration-nodes'])

    @spline_calibration_nodes.setter
    def spline_calibration_nodes(self, value: int):
        """Set the number of spline calibration nodes.

        Args:
            value (int): Number of spline calibration nodes.
        """
        self['spline-calibration-nodes'] = value

    @property
    def spline_calibration_amplitude_uncertainty_dict(self) -> Union[dict, None]:
        """Get the spline calibration amplitude uncertainty dictionary.

        Returns:
            dict: Spline calibration amplitude uncertainty dictionary.
        """
        return convert_string_to_dict(self['spline-calibration-amplitude-uncertainty-dict'])

    @spline_calibration_amplitude_uncertainty_dict.setter
    def spline_calibration_amplitude_uncertainty_dict(self, value: dict):
        """Set the spline calibration amplitude uncertainty dictionary.

        Args:
            value (dict): Spline calibration amplitude uncertainty dictionary.
        """
        self['spline-calibration-amplitude-uncertainty-dict'] = value

    @property
    def spline_calibration_phase_uncertainty_dict(self) -> Union[dict, None]:
        """Get the spline calibration phase uncertainty dictionary.

        Returns:
            dict: Spline calibration phase uncertainty dictionary.
        """
        return convert_string_to_dict(self['spline-calibration-phase-uncertainty-dict'])

    @spline_calibration_phase_uncertainty_dict.setter
    def spline_calibration_phase_uncertainty_dict(self, value: dict):
        """Set the spline calibration phase uncertainty dictionary.

        Args:
            value (dict): Spline calibration phase uncertainty dictionary.
        """
        self['spline-calibration-phase-uncertainty-dict'] = value

    @property
    def calibration_prior_boundary(self) -> Union[str, None]:
        """Get the calibration prior boundary.

        Returns:
            str: Calibration prior boundary.
        """
        return self['calibration-prior-boundary']

    @calibration_prior_boundary.setter
    def calibration_prior_boundary(self, value: str):
        """Set the calibration prior boundary.

        Args:
            value (str): Calibration prior boundary.
        """
        self['calibration-prior-boundary'] = value

    @property
    def calibration_correction_type(self) -> Union[str, None]:
        """Get the calibration correction type.

        Returns:
            str: Calibration correction type.
        """
        return self['calibration-correction-type']

    @calibration_correction_type.setter
    def calibration_correction_type(self, value: str):
        """Set the calibration correction type.

        Args:
            value (str): Calibration correction type.
        """
        self['calibration-correction-type'] = value

    @property
    def ignore_gwpy_data_quality_check(self) -> Union[bool, None]:
        """Whether to ignore gwpy data quality check.

        Returns:
            bool: Whether to ignore gwpy data quality check.
        """
        return convert_string_to_bool(self['ignore-gwpy-data-quality-check'])

    @ignore_gwpy_data_quality_check.setter
    def ignore_gwpy_data_quality_check(self, value: bool):
        """Set whether to ignore gwpy data quality check.

        Args:
            value (bool): Whether to ignore gwpy data quality check.
        """
        self['ignore-gwpy-data-quality-check'] = value

    @property
    def gps_tuple(self) -> Union[tuple, None]:
        """Get the gps tuple.

        Returns:
            tuple: gps tuple.
        """
        return convert_string_to_tuple(self['gps-tuple'])

    @gps_tuple.setter
    def gps_tuple(self, value: tuple):
        """Set the gps tuple.

        Args:
            value (tuple): gps tuple.
        """
        self['gps-tuple'] = value

    @property
    def gps_file(self) -> Union[str, None]:
        """Get the gps file.

        Returns:
            str: gps file.
        """
        return self['gps-file']

    @gps_file.setter
    def gps_file(self, value: str):
        """Set the gps file.

        Args:
            value (str): gps file.
        """
        self['gps-file'] = value

    @property
    def timeslide_file(self) -> Union[str, None]:
        """Get the timeslide file.

        Returns:
            str: timeslide file.
        """
        return self['timeslide-file']

    @timeslide_file.setter
    def timeslide_file(self, value: str):
        """Set the timeslide file.

        Args:
            value (str): timeslide file.
        """
        self['timeslide-file'] = value

    @property
    def timeslide_dict(self) -> Union[dict, None]:
        """Get the timeslide dictionary.

        Returns:
            dict: timeslide dictionary.
        """
        return convert_string_to_dict(self['timeslide-dict'])

    @timeslide_dict.setter
    def timeslide_dict(self, value: dict):
        """Set the timeslide dictionary.

        Args:
            value (dict): timeslide dictionary.
        """
        self['timeslide-dict'] = value

    @property
    def trigger_time(self) -> Union[float, None]:
        """Get the trigger time.

        Returns:
            float: Trigger time.
        """
        return convert_string_to_float(self['trigger-time'])

    @trigger_time.setter
    def trigger_time(self, value: float):
        """Set the trigger time.

        Args:
            value (float): Trigger time.
        """
        self['trigger-time'] = value

    @property
    def n_simulation(self) -> Union[int, None]:
        """Get the number of simulations.

        Returns:
            int: Number of simulations.
        """
        return convert_string_to_int(self['n-simulation'])

    @n_simulation.setter
    def n_simulation(self, value: int):
        """Set the number of simulations.

        Args:
            value (int): Number of simulations.
        """
        self['n-simulation'] = int(value)

    @property
    def data_dict(self) -> Union[dict, None]:
        """Get the data dictionary.

        Returns:
            dict: Data dictionary.
        """
        return convert_string_to_dict(self['data-dict'])

    @data_dict.setter
    def data_dict(self, value: dict):
        """Set the data dictionary.

        Args:
            value (dict): Data dictionary.
        """
        self['data-dict'] = value

    @property
    def data_format(self) -> Union[str, None]:
        """Get the data format.

        Returns:
            str: Data format.
        """
        return self['data-format']

    @data_format.setter
    def data_format(self, value: str):
        """Set the data format.

        Args:
            value (str): Data format.
        """
        self['data-format'] = value

    @property
    def allow_tape(self) -> Union[bool, None]:
        """Allow tape.

        Returns:
            bool: Allow taple.
        """
        return convert_string_to_bool(self['allow-tape'])

    @allow_tape.setter
    def allow_tape(self, value: bool):
        """Allow taple.

        Args:
            value (bool): Allow taple.
        """
        self['allow-tape'] = value

    @property
    def channel_dict(self) -> Union[dict, None]:
        """Get the channel dictionary.

        Returns:
            dict: Channel dictionary.
        """
        return convert_string_to_dict(self['channel-dict'])

    @channel_dict.setter
    def channel_dict(self, value: dict):
        """Set the channel dictionary.

        Args:
            value (dict): Channel dictionary.
        """
        self['channel-dict'] = value

    @property
    def frame_type_dict(self) -> Union[dict, None]:
        """Get the frame type dictionary.

        Returns:
            dict: Frame type dictionary.
        """
        return convert_string_to_dict(self['frame-type-dict'])

    @frame_type_dict.setter
    def frame_type_dict(self, value: dict):
        """Set the frame type dictionary.

        Args:
            value (dict): Frame type dictionary.
        """
        self['frame-type-dict'] = value

    @property
    def data_find_url(self) -> Union[str, None]:
        """Get the data find url.

        Returns:
            str: Data find url.
        """
        return self['data-find-url']

    @data_find_url.setter
    def data_find_url(self, value: str):
        """Set the data find url.

        Args:
            value (str): Data find url.
        """
        self['data-find-url'] = value

    @property
    def data_find_urltype(self) -> Union[str, None]:
        """Get the data find url type.

        Returns:
            str: Data find url type.
        """
        return self['data-find-urltype']

    @data_find_urltype.setter
    def data_find_urltype(self, value: str):
        """Set the data find url type.

        Args:
            value (str): Data find url type.
        """
        self['data-find-urltype'] = value

    @property
    def gaussian_noise(self) -> Union[bool, None]:
        """Whether to set Gaussian noise.

        Returns:
            bool: Whether or not to set Gaussian noise.
        """
        return convert_string_to_bool(self['gaussian-noise'])

    @gaussian_noise.setter
    def gaussian_noise(self, value: bool):
        """Set whether to set Gaussian noise.

        Args:
            value (bool): Set whether to set Gaussian noise.
        """
        self['gaussian-noise'] = value

    @property
    def zero_noise(self) -> Union[bool, None]:
        """Whether to set zero noise.

        Returns:
            bool: Whether to set zero noise.
        """
        return convert_string_to_bool(self['zero-noise'])

    @zero_noise.setter
    def zero_noise(self, value: bool):
        """Whether to set zero noise.

        Args:
            value (bool): Whether to set zero noise.
        """
        self['zero-noise'] = value

    @property
    def detectors(self) -> Union[list, None]:
        """Get list of detectors.

        Returns:
            list: List of detectors.
        """
        return convert_string_to_list(self['detectors'])

    @detectors.setter
    def detectors(self, value: list):
        """Set list of detectors.

        Args:
            value (list): List of detectors.
        """
        self['detectors'] = value

    @property
    def duration(self) -> Union[float, None]:
        """Get the duration.

        Returns:
            float: Duration.
        """
        return convert_string_to_float(self['duration'])

    @duration.setter
    def duration(self, value: float):
        """Set the duration.

        Args:
            value (float): Duration.
        """
        self['duration'] = value

    @property
    def generation_seed(self) -> Union[int, None]:
        """Get the generation seed.

        Returns:
            int: Generation seed.
        """
        return convert_string_to_int(self['generation-seed'])

    @generation_seed.setter
    def generation_seed(self, value: int):
        """Set the generation seed.

        Args:
            value (int): Generation seed.
        """
        self['generation-seed'] = value

    @property
    def psd_dict(self) -> Union[dict, None]:
        """Get the psd dictionary.

        Returns:
            dict: psd dictionary.
        """
        return convert_string_to_dict(self['psd-dict'])

    @psd_dict.setter
    def psd_dict(self, value: dict):
        """Set the psd dictionary.

        Args:
            value (dict): psd dictionary.
        """
        self['psd-dict'] = value

    @property
    def psd_fractional_overlap(self) -> Union[float, None]:
        """Get the psd fractional overlap.

        Returns:
            float: psd fractional overlap.
        """
        return convert_string_to_float(self['psd-fractional-overlap'])

    @psd_fractional_overlap.setter
    def psd_fractional_overlap(self, value: float):
        """Set the psd fractional overlap.

        Args:
            value (float): psd fractional overlap.
        """
        self['psd-fractional-overlap'] = value

    @property
    def post_trigger_duration(self) -> Union[float, None]:
        """Get the post trigger duration.

        Returns:
            float: Post trigger duration.
        """
        return convert_string_to_float(self['post-trigger-duration'])

    @post_trigger_duration.setter
    def post_trigger_duration(self, value: float):
        """Set the post trigger duration.

        Args:
            value (float): Post trigger duration.
        """
        self['post-trigger-duration'] = value

    @property
    def sampling_frequency(self) -> Union[float, None]:
        """Get the sampling frequency.

        Returns:
            float: Sampling frequency.
        """
        return convert_string_to_float(self['sampling-frequency'])

    @sampling_frequency.setter
    def sampling_frequency(self, value: float):
        """Set the sampling frequency.

        Args:
            value (float): Sampling frequency.
        """
        self['sampling-frequency'] = value

    @property
    def psd_length(self) -> Union[int, None]:
        """Get the psd length.

        Returns:
            int: psd length.
        """
        return convert_string_to_int(self['psd-length'])

    @psd_length.setter
    def psd_length(self, value: int):
        """Set the psd length.

        Args:
            value (int): psd length.
        """
        self['psd-length'] = value

    @property
    def psd_maximum_duration(self) -> Union[float, None]:
        """Get the psd maximum duration.

        Returns:
            float: psd maximum duration.
        """
        return convert_string_to_float(self['psd-maximum-duration'])

    @psd_maximum_duration.setter
    def psd_maximum_duration(self, value: float):
        """Set the psd maximum duration.

        Args:
            value (float): psd maximum duration.
        """
        self['psd-maximum-duration'] = value

    @property
    def psd_method(self) -> Union[str, None]:
        """Get the psd method.

        Returns:
            str: psd method.
        """
        return self['psd-method']

    @psd_method.setter
    def psd_method(self, value: str):
        """Set the psd method.

        Args:
            value (str): psd method.
        """
        self['psd-method'] = value

    @property
    def psd_start_time(self) -> Union[float, None]:
        """Get the psd start time.

        Returns:
            float: psd start time.
        """
        return convert_string_to_float(self['psd-start-time'])

    @psd_start_time.setter
    def psd_start_time(self, value: float):
        """Set the psd start time.

        Args:
            value (float): psd start time.
        """
        self['psd-start-time'] = value

    @property
    def maximum_frequency(self) -> Union[dict, None]:
        """Get the maximum frequency.

        Returns:
            float: Maximum frequency.
        """
        return convert_string_to_dict(self['maximum-frequency'])

    @maximum_frequency.setter
    def maximum_frequency(self, value: dict):
        """Set the maximum frequency.

        Args:
            value (dict): Maximum frequency.
        """
        self['maximum-frequency'] = value

    @property
    def minimum_frequency(self) -> Union[dict, None]:
        """Get the minimum frequency.

        Returns:
            dict: Minimum frequency.
        """
        return convert_string_to_dict(self['minimum-frequency'])

    @minimum_frequency.setter
    def minimum_frequency(self, value: dict):
        """Set the minimum frequency.

        Args:
            value (dict): Minimum frequency.
        """
        self['minimum-frequency'] = value

    @property
    def tukey_roll_off(self) -> Union[float, None]:
        """Get the tukey roll off.

        Returns:
            float: tukey roll off.
        """
        return convert_string_to_float(self['tukey-roll-off'])

    @tukey_roll_off.setter
    def tukey_roll_off(self, value: float):
        """Set the tukey roll off.

        Args:
            value (float): tukey roll off.
        """
        self['tukey-roll-off'] = value

    @property
    def resampling_method(self) -> Union[str, None]:
        """Get the resampling method.

        Returns:
            str: Resampling method.
        """
        return self['resampling-method']

    @resampling_method.setter
    def resampling_method(self, value: str):
        """Set the resampling method.

        Args:
            value (str): Resampling method.
        """
        self['resampling-method'] = value

    @property
    def injection(self) -> Union[bool, None]:
        """Whether to do injection.

        Returns:
            bool: Whether to do injeciton.
        """
        return convert_string_to_bool(self['injection'])

    @injection.setter
    def injection(self, value: bool):
        """Set whether to do injection.

        Args:
            value (bool): Whether to do injection.
        """
        self['injection'] = value

    @property
    def injection_dict(self) -> Union[dict, None]:
        """Get the injection dictionary.

        Returns:
            dict: Injection dictionary.
        """
        return convert_string_to_dict(self['injection-dict'])

    @injection_dict.setter
    def injection_dict(self, value: dict):
        """Set the injection dictionary.

        Args:
            value (dict: Injection dictionary.
        """
        self['injection-dict'] = value

    @property
    def injection_file(self) -> Union[str, None]:
        """Get the injection file.

        Returns:
            str: Injection file.
        """
        return self['injection-file']

    @injection_file.setter
    def injection_file(self, value: str):
        """Set the injection file.

        Args:
            value (str): Injection file.
        """
        self['injection-file'] = value

    @property
    def injection_numbers(self) -> Union[list, None]:
        """Get the specific injection rows from the injection file.

        Returns:
            list: Specific injection rows from the injection file.
        """
        return convert_string_to_list(self['injection-numbers'])

    @injection_numbers.setter
    def injection_numbers(self, value: list):
        """Set the specific injection rows from the injeciton file.

        Args:
            value (list): Specific injection rows from the injection file.
        """
        self['injection-numbers'] = value

    @property
    def injection_waveform_approximant(self) -> Union[str, None]:
        """Get the injection waveform approximant.

        Returns:
            str: Injection waveform approximant.
        """
        return self['injection-waveform-approximant']

    @injection_waveform_approximant.setter
    def injection_waveform_approximant(self, value: str):
        """Set the injection waveform approximant.

        Args:
            value (str): Injection waveform approximant.
        """
        self['injection-waveform-approximant'] = value

    @property
    def injection_frequency_domain_source_model(self) -> Union[str, None]:
        """Get the injection frequency domain source model.

        Returns:
            str: Injection frequency domain source model.
        """
        return self['injection-frequency-domain-source-model']

    @injection_frequency_domain_source_model.setter
    def injection_frequency_domain_source_model(self, value: str):
        """Set the injection frequency domain source model.

        Args:
            value (str): Injection frequency domain source model.
        """
        self['injection-frequency-domain-source-model'] = value

    @property
    def injection_waveform_arguments(self) -> Union[dict, None]:
        """Get the injection waveform arguments.

        Returns:
            dict: Injection waveform arguments.
        """
        return convert_string_to_dict(self['injection-waveform-arguments'])

    @injection_waveform_arguments.setter
    def injection_waveform_arguments(self, value: dict):
        """Set the injection waveform arguments.

        Args:
            value (dict): Injection waveform arguments.
        """
        self['injection-waveform-arguments'] = value

    @property
    def accounting(self) -> Union[str, None]:
        """Get the accounting.

        Returns:
            str: Accounting.
        """
        return self['accounting']

    @accounting.setter
    def accounting(self, value: str):
        """Set the accounting.

        Returns:
            str: Accounting.
        """
        self['accounting'] = value

    @property
    def accounting_user(self) -> Union[str, None]:
        """Get the accounting user.

        Returns:
            str: Accounting user.
        """
        return self['accounting-user']

    @accounting_user.setter
    def accounting_user(self, value: str):
        """Set the accounting user.

        Args:
            value (str): Accounting user.
        """
        self['accounting-user'] = value

    @property
    def label(self) -> Union[str, None]:
        """Get the label.

        Returns:
            str: Label.
        """
        return self['label']

    @label.setter
    def label(self, value: str):
        """Set the label.

        Args:
            value (str): Label
        """
        self['label'] = value

    @property
    def local(self) -> Union[bool, None]:
        """Whether to run locally.

        Returns:
            bool: Whether to run locally.
        """
        return convert_string_to_bool(self['local'])

    @local.setter
    def local(self, value: bool):
        """Set whether to run locally.

        Args:
            value (bool): Whether to run locally.
        """
        self['local'] = value

    @property
    def local_generation(self) -> Union[bool, None]:
        """Whether to generate data locally.

        Returns:
            bool: Whether to generate data locally.
        """
        return convert_string_to_bool(self['local-generation'])

    @local_generation.setter
    def local_generation(self, value: bool):
        """Set whether to generate data locally.

        Args:
            value (bool): Whether to generate data locally.
        """
        self['local-generation'] = value

    @property
    def generation_pool(self) -> Union[str, None]:
        """Get the generation pool.

        Returns:
            str: Generation bool.
        """
        return self['generation-pool']

    @generation_pool.setter
    def generation_pool(self, value: str):
        """Set the generation pool.

        Args:
            value (str): Generation pool.
        """
        self['generation-pool'] = value

    @property
    def local_plot(self) -> Union[bool, None]:
        """Get whether to generate plot locally.

        Returns:
            bool: Whether to generate plot locally.
        """
        return convert_string_to_bool(self['local-plot'])

    @local_plot.setter
    def local_plot(self, value: bool):
        """Set whether to generate plot locally.

        Args:
            value (bool): Whether to generate plot locally.
        """
        self['local-plot'] = value

    @property
    def outdir(self) -> Union[str, None]:
        """Get the output directory.

        Returns:
            str: Output directory.
        """
        return self['outdir']

    @outdir.setter
    def outdir(self, value: str):
        """Set the output directory.

        Args:
            value (str): Output directory.
        """
        self['outdir'] = value

    @property
    def overwrite_outdir(self) -> Union[bool, None]:
        """Get whether to overwrite output directory.

        Returns:
            bool: Whether to overwrite output directory.
        """
        return convert_string_to_bool(self['overwrite-outdir'])

    @overwrite_outdir.setter
    def overwrite_outdir(self, value: bool):
        """Set whether to overwrite output directory.

        Args:
            value (bool): Whether to overwrite output directory.
        """
        self['overwrite-outdir'] = value

    @property
    def periodic_restart_time(self) -> Union[int, None]:
        """Get the periodic restart time.

        Returns:
            int: Periodic restart time.
        """
        return convert_string_to_int(self['periodic-restart-time'])

    @periodic_restart_time.setter
    def periodic_restart_time(self, value: int):
        """Set the periodic restart time.

        Args:
            value (int): Periodic restart time.
        """
        self['periodic-restart-time'] = value

    @property
    def request_disk(self) -> Union[float, None]:
        """Get the request disk.

        Returns:
            float: Request disk.
        """
        return convert_string_to_float(self['request-disk'])

    @request_disk.setter
    def request_disk(self, value: float):
        """Set the request disk.

        Args:
            value (float): Request disk.
        """
        self['request-disk'] = value

    @property
    def request_memory(self) -> Union[float, None]:
        """Get the request memory.

        Returns:
            float: Request memory.
        """
        return convert_string_to_float(self['request-memory'])

    @request_memory.setter
    def request_memory(self, value: float):
        """Set the request memory.

        Args:
            value (float): Request memory.
        """
        self['request-memory'] = value

    @property
    def request_memory_generation(self) -> Union[float, None]:
        """Get the request memory for data generation.

        Returns:
            float: Request memory for data generation.
        """
        return convert_string_to_float(self['request-memory-generation'])

    @request_memory_generation.setter
    def request_memory_generation(self, value: float):
        """Set the request memory for data generation.

        Args:
            value (float): Request memory for data generation.
        """
        self['request-memory-generation'] = value

    @property
    def request_cpus(self) -> Union[int, None]:
        """Get the number of request cpus.

        Returns:
            int: Number of request cpus.
        """
        return convert_string_to_int(self['request-cpus'])

    @request_cpus.setter
    def request_cpus(self, value: int):
        """Set the number of request cpus.

        Args:
            value (int): Number of request cpis.
        """
        self['request-cpus'] = value

    @property
    def conda_env(self) -> Union[str, None]:
        """Get the conda environment.

        Returns:
            str: Conda environment.
        """
        return self['conda-env']

    @conda_env.setter
    def conda_env(self, value: str):
        """Set the conda environment.

        Args:
            value (str): Conda environment.
        """
        self['conda-env'] = value

    @property
    def scheduler(self) -> Union[str, None]:
        """Get the scheduler.

        Returns:
            str: Scheduler.
        """
        return self['scheduler']

    @scheduler.setter
    def scheduler(self, value: str):
        """Set the scheduler.

        Args:
            value (str): Scheduler.
        """
        self['scheduler'] = value

    @property
    def scheduler_args(self) -> Union[str, None]:
        """Get the scheduler arguments.

        Returns:
            str: Scheduler arguments.
        """
        return self['scheduler-args']

    @scheduler_args.setter
    def scheduler_args(self, value: str):
        """Set the scheduler arguments.

        Args:
            value (str): Sceduler arguments.
        """
        self['scheduler-args'] = value

    @property
    def scheduler_module(self) -> Union[list, None]:
        """Get the scheduler module.

        Returns:
            list: Scheduler module.
        """
        return convert_string_to_list(self['scheduler-module'])

    @scheduler_module.setter
    def scheduler_module(self, value: list):
        """Set the scheduler module.

        Args:
            value (list): Scheduler module.
        """
        self['scheduler-module'] = value

    @property
    def scheduler_env(self) -> Union[str, None]:
        """Get the scheduler environment.

        Returns:
            str: Scheduler environment.
        """
        return self['scheduler-env']

    @scheduler_env.setter
    def scheduler_env(self, value: str):
        """Set the scheduler environment.

        Args:
            value (str): Scheduler environment.
        """
        self['scheduler-env'] = value

    @property
    def scheduler_analysis_time(self) -> Union[str, None]:
        """Get the scheduler analysis time.

        Returns:
            str: Scheduler analysis time.
        """
        return self['scheduler-analysis-time']

    @scheduler_analysis_time.setter
    def scheduler_analysis_time(self, value: str):
        """Set the scheduler analysis time.

        Args:
            value (str): Scheduler analysis time.
        """
        self['scheduler-analysis-time'] = value

    @property
    def submit(self) -> Union[bool, None]:
        """Get whether to submit.

        Returns:
            bool: Whether to submit.
        """
        return convert_string_to_bool[self['submit']]

    @submit.setter
    def submit(self, value: bool):
        """Set whether to submit.

        Args:
            value (bool): Whether to submit.
        """
        self['submit'] = value

    @property
    def condor_job_priority(self) -> Union[int, None]:
        """Get the condor job priority.

        Returns:
            int: Condor job priority.
        """
        return convert_string_to_int(self['condor-job-priority'])

    @condor_job_priority.setter
    def condor_job_priority(self, value: int):
        """Set the condor job priority.

        Args:
            value (int): Condor job priority.
        """
        self['condor-job-priority'] = value

    @property
    def transfer_files(self) -> Union[bool, None]:
        """Get whether to transfer files.

        Returns:
            bool: Whether to transfer files.
        """
        return convert_string_to_bool(self['transfer-files'])

    @transfer_files.setter
    def transfer_files(self, value: bool):
        """Set whether to transfer files.

        Args:
            value (bool): Whether to transfer files.
        """
        self['transfer-files'] = value

    @property
    def additional_transfer_paths(self) -> Union[str, None]:
        """Get additional transfer paths.

        Returns:
            str: Additional transfer paths.
        """
        return self['additional-transfer-paths']

    @additional_transfer_paths.setter
    def additional_transfer_paths(self, value: str):
        """Set additional transfer paths.

        Args:
            value (str): Additional transfer paths.
        """
        self['additional-transfer-paths'] = value

    @property
    def environment_variables(self) -> Union[dict, None]:
        """Get the envirionment variables.

        Returns:
            dict: Environment variables.
        """
        return convert_string_to_dict(self['environment-variables'])

    @environment_variables.setter
    def environment_variables(self, value: dict):
        """Set the environment variables.

        Args:
            value (dict): Environment variables.
        """
        self['environment-variables'] = value

    @property
    def getenv(self) -> Union[list, None]:
        """Get the list of environment variables.

        Returns:
            list: List of environment variables.
        """
        return convert_string_to_list(self['getenv'])

    @getenv.setter
    def getenv(self, value: list):
        """Set the list of environment variables.

        Args:
            value (list): List of environment variables.
        """
        self['getenv'] = value

    @property
    def disable_hdf5_locking(self) -> Union[bool, None]:
        """Get whether to disable hdf5 locking.

        Returns:
            bool: Whether to disable hdf5 locking.
        """
        return convert_string_to_bool(self['disable-hdf5-locking'])

    @disable_hdf5_locking.setter
    def disable_hdf5_locking(self, value: bool):
        """Set whether to disable hdf5 locking.

        Args:
            value (bool): Whether to disable hdf5 locking.
        """
        self['disable-hdf5-locking'] = value

    @property
    def log_directory(self) -> Union[str, None]:
        """Get the log directory.

        Returns:
            str: Log directory.
        """
        return self['log-directory']

    @log_directory.setter
    def log_directory(self, value: str):
        """Set the log directory.

        Args:
            value (str): Log directory.
        """
        self['log-directory'] = value

    @property
    def osg(self) -> Union[bool, None]:
        """Get whether to run on osg.

        Returns:
            bool: Whether to run on osg.
        """
        return convert_string_to_bool(self['osg'])

    @osg.setter
    def osg(self, value: bool):
        """Set whether to run on osg.

        Args:
            value (bool): Whether to run on osg.
        """
        self['osg'] = value

    @property
    def desired_sites(self) -> Union[str, None]:
        """Get the desired sites.

        Returns:
            str: Desired sites.
        """
        return self['desired-sites']

    @desired_sites.setter
    def desired_sites(self, value: str):
        """Set the desired sites.

        Args:
            value (str): Desired sites.
        """
        self['desired-sites'] = value

    @property
    def analysis_executable(self) -> Union[str, None]:
        """Get the analysis executable.

        Returns:
            str: Analysis executable.
        """
        return self['analysis-executable']

    @analysis_executable.setter
    def analysis_executable(self, value: str):
        """Set the analysis executable.

        Args:
            value (str): Analysis executable.
        """
        self['analysis-executable'] = value

    @property
    def analysis_executable_parser(self) -> Union[str, None]:
        """Get the analysis executable parser.

        Returns:
            str: Analysis executable parser.
        """
        return self['analysis-executable-parser']

    @analysis_executable_parser.setter
    def analysis_executable_parser(self, value: str):
        """Set the analysis executable parser.

        Args:
            value (str): Analysis executable parser.
        """
        self['analysis-executable-parser'] = value

    @property
    def scitoken_issuer(self) -> Union[str, None]:
        """Get the scitoken issuer.

        Returns:
            str: Scitoken issuer.
        """
        return self['scitoken-issuer']

    @scitoken_issuer.setter
    def scitoken_issuer(self, value: str):
        """Set the scitoken issuer.

        Args:
            value (str): Scitoken issuer.
        """
        self['scitoken-issuer'] = value

    @property
    def reference_frame(self) -> Union[str, None]:
        """Get the reference frame.

        Returns:
            str: Reference frame.
        """
        return self['reference-frame']

    @reference_frame.setter
    def reference_frame(self, value: str):
        """Set the reference frame.

        Args:
            value (str): Reference frame.
        """
        self['reference-frame'] = value

    @property
    def time_reference(self) -> Union[str, None]:
        """Get the time reference.

        Returns:
            str: Time reference.
        """
        return self['time-reference']

    @time_reference.setter
    def time_reference(self, value: str):
        """Set the time reference.

        Args:
            value (str): Time reference.
        """
        self['time-reference'] = value

    @property
    def extra_likelihood_kwargs(self) -> Union[dict, None]:
        """Get the extra likelihood kwargs.

        Returns:
            dict: Extra likelihood kwargs.
        """
        return convert_string_to_dict(self['extra-likelihood-kwargs'])

    @extra_likelihood_kwargs.setter
    def extra_likelihood_kwargs(self, value: dict):
        """Set the extra likelihood kwargs.

        Args:
            value (dict): Extra likelihood kwargs.
        """
        self['extra-likelihood-kwargs'] = value

    @property
    def likelihood_type(self) -> Union[str, None]:
        """Get the likelihood type.

        Returns:
            str: Likelihood type.
        """
        return self['likelihood-type']

    @likelihood_type.setter
    def likelihood_type(self, value: str):
        """Set the likelihood type.

        Args:
            value (str): Likelihood type.
        """
        self['likelihood-type'] = value

    @property
    def polarization_modes(self) -> Union[list, None]:
        """Get the list of polarization modes.

        Returns:
            list: A list of polarization modes.
        """
        return convert_string_to_list(self['polarization-modes'])

    @polarization_modes.setter
    def polarization_modes(self, value: list):
        """Set the list of polarization modes.

        Args:
            value (list): A list of polarization modes.
        """
        self['polarization-modes'] = value

    @property
    def polarization_basis(self) -> Union[list, None]:
        """Get the list of polarization basis.

        Returns:
            list: A list of polarization modes.
        """
        return convert_string_to_list(self['polarization-basis'])

    @polarization_basis.setter
    def polarization_basis(self, value: list):
        """Set the list of polarization basis.

        Args:
            value (list): A list of polarization basis.
        """
        self['polarization-basis'] = value

    @property
    def wavelet_frequency_resolution(self) -> Union[float, None]:
        """Get the wavelet frequency resolution.

        Returns:
            float: Wavelet frequency resolution.
        """
        return convert_string_to_float(self['wavelet-frequency-resolution'])

    @wavelet_frequency_resolution.setter
    def wavelet_frequency_resolution(self, value: float):
        """Set the wavelet frequency resolution.

        Args:
            value (float): Wavelet frequency resolution.
        """
        self['wavelet-frequency-resolution'] = value

    @property
    def wavelet_nx(self) -> Union[float, None]:
        """Get the wavelet nx.

        Returns:
            float: Wavelet nx.
        """
        return convert_string_to_float(self['wavelet-nx'])

    @wavelet_nx.setter
    def wavelet_nx(self, value: float):
        """Set the wavelet nx.

        Args:
            value (float): Wavelet nx.
        """
        self['wavelet-nx'] = value
