from bilby_pipe.utils import (parse_args,
                              get_command_line_arguments,
                              get_outdir_name,
                              tcolors)
from bilby_pipe.main import write_complete_config_file
from bilby_pipe.input import Input
from bilby_pipe.main import MainInput as BilbyMainInput
import bilby_pipe.utils
import importlib
from .parser import create_nullpol_parser
from .input import Input
from .. import log_version_information
from ..utility import logger
from ..job_creation import generate_dag


bilby_pipe.utils.logger = logger

class MainInput(BilbyMainInput, Input):
    def __init__(self, args, unknown_args, perform_checks=True):
        Input.__init__(self, args, unknown_args, print_msg=False)

        self.known_args = args
        self.unknown_args = unknown_args
        self.ini = args.ini
        self.submit = args.submit
        self.condor_job_priority = args.condor_job_priority
        self.create_summary = args.create_summary
        self.scitoken_issuer = args.scitoken_issuer

        self.outdir = args.outdir
        self.label = args.label
        self.log_directory = args.log_directory
        self.accounting = args.accounting
        self.accounting_user = args.accounting_user
        self.sampler = args.sampler
        self.sampling_seed = args.sampling_seed
        self.detectors = args.detectors
        self.data_dict = args.data_dict
        self.channel_dict = args.channel_dict
        self.frame_type_dict = args.frame_type_dict
        self.data_find_url = args.data_find_url
        self.data_find_urltype = args.data_find_urltype
        self.n_parallel = args.n_parallel
        self.transfer_files = args.transfer_files
        self.additional_transfer_paths = args.additional_transfer_paths
        self.osg = args.osg
        self.desired_sites = args.desired_sites
        self.analysis_executable = args.analysis_executable
        self.analysis_executable_parser = args.analysis_executable_parser
        self.result_format = args.result_format
        self.final_result = args.final_result
        self.final_result_nsamples = args.final_result_nsamples

        self.webdir = args.webdir
        self.email = args.email
        self.notification = args.notification
        self.queue = args.queue
        self.existing_dir = args.existing_dir

        self.scheduler = args.scheduler
        self.scheduler_args = args.scheduler_args
        self.scheduler_module = args.scheduler_module
        self.scheduler_env = args.scheduler_env
        self.scheduler_analysis_time = args.scheduler_analysis_time
        self.disable_hdf5_locking = args.disable_hdf5_locking
        self.environment_variables = args.environment_variables
        self.getenv = args.getenv

        self.waveform_approximant = args.waveform_approximant

        self.time_reference = args.time_reference
        self.reference_frame = args.reference_frame
        self.likelihood_type = args.likelihood_type
        self.duration = args.duration
        self.prior_file = args.prior_file
        self.prior_dict = args.prior_dict
        self.default_prior = args.default_prior
        self.minimum_frequency = args.minimum_frequency
        self.enforce_signal_duration = args.enforce_signal_duration

        self.run_local = args.local
        self.generation_pool = args.generation_pool
        self.local_generation = args.local_generation
        self.local_plot = args.local_plot

        self.post_trigger_duration = args.post_trigger_duration

        self.ignore_gwpy_data_quality_check = args.ignore_gwpy_data_quality_check
        self.trigger_time = args.trigger_time
        self.deltaT = args.deltaT
        self.gps_tuple = args.gps_tuple
        self.gps_file = args.gps_file
        self.timeslide_file = args.timeslide_file
        self.gaussian_noise = args.gaussian_noise
        self.zero_noise = args.zero_noise
        self.n_simulation = args.n_simulation

        self.injection = args.injection
        self.injection_numbers = args.injection_numbers
        self.injection_file = args.injection_file
        self.injection_dict = args.injection_dict
        self.injection_waveform_arguments = args.injection_waveform_arguments
        self.injection_waveform_approximant = args.injection_waveform_approximant
        self.injection_frequency_domain_source_model = (
            args.injection_frequency_domain_source_model
        )
        self.generation_seed = args.generation_seed

        self.request_disk = args.request_disk
        self.request_memory = args.request_memory
        self.request_memory_generation = args.request_memory_generation
        self.request_cpus = args.request_cpus
        self.sampler_kwargs = args.sampler_kwargs
        self.mpi_samplers = ["pymultinest"]
        self.use_mpi = (self.sampler in self.mpi_samplers) and (self.request_cpus > 1)

        # Set plotting options when need the plot node
        self.plot_node_needed = False
        for plot_attr in [
            "calibration",
            "corner",
            "marginal",
            "skymap",
            "waveform",
        ]:
            attr = f"plot_{plot_attr}"
            setattr(self, attr, getattr(args, attr))
            if getattr(self, attr):
                self.plot_node_needed = True

        # Set all other plotting options
        for plot_attr in [
            "trace",
            "data",
            "injection",
            "spectrogram",
            "format",
        ]:
            attr = f"plot_{plot_attr}"
            setattr(self, attr, getattr(args, attr))

        self.postprocessing_executable = args.postprocessing_executable
        self.postprocessing_arguments = args.postprocessing_arguments
        self.single_postprocessing_executable = args.single_postprocessing_executable
        self.single_postprocessing_arguments = args.single_postprocessing_arguments

        self.summarypages_arguments = args.summarypages_arguments

        self.psd_dict = args.psd_dict
        self.psd_maximum_duration = args.psd_maximum_duration
        self.psd_length = args.psd_length
        self.psd_fractional_overlap = args.psd_fractional_overlap
        self.psd_start_time = args.psd_start_time
        self.spline_calibration_envelope_dict = args.spline_calibration_envelope_dict

        if perform_checks:
            self.check_source_model(args)
            self.check_calibration_prior_boundary(args)
            self.check_cpu_parallelisation()
            if self.injection:
                self.check_injection()

        self.extra_lines = []
        self.requirements = []

    @property
    def analysis_executable(self):
        return self._analysis_executable

    @analysis_executable.setter
    def analysis_executable(self, analysis_executable):
        if analysis_executable:
            self._analysis_executable = analysis_executable
        else:
            self._analysis_executable = "nullpol_pipe_analysis"

def create_main_parser():
    _nullpol_pipe_doc = """
    nullpol_pipe is a command line tools for taking user input (as command line
    arguments or an ini file) and creating DAG files for submitting nullpol parameter
    estimation jobs. To get started, write an ini file `config.ini` and run

    .. code-block:: console

        $ bilby_pipe config.ini

    Instruction for how to submit the job are printed in a log message. You can
    also specify extra arguments from the command line, e.g.

    .. code-block:: console

        $ nullpol_pipe config.ini --submit

    will build and submit the job.
    """
    return create_nullpol_parser(top_level=True)

def main():
    """Top-level interface for nullpol_pipe"""
    parser = create_main_parser()
    args, unknown_args = parse_args(get_command_line_arguments(), parser)

    if args.analysis_executable_parser is not None:
        # Alternative parser requested, reload args
        module = ".".join(args.analysis_executable_parser.split(".")[:-1])
        function = args.analysis_executable_parser.split(".")[-1]
        parser = getattr(importlib.import_module(module), function)()
        args, unknown_args = parse_args(get_command_line_arguments(), parser)

    # Check and sort outdir
    args.outdir = args.outdir.replace("'", "").replace('"', "")
    if args.overwrite_outdir is False:
        args.outdir = get_outdir_name(args.outdir)

    log_version_information()
    inputs = MainInput(args, unknown_args)
    write_complete_config_file(parser, args, inputs, input_cls=MainInput)
    generate_dag(inputs)

    if len(unknown_args) > 0:
        msg = [tcolors.WARNING, f"Unrecognized arguments {unknown_args}", tcolors.END]
        logger.warning(" ".join(msg))
