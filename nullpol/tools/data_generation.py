from bilby_pipe.input import Input
from bilby_pipe.data_generation import DataGenerationInput as BilbyDataGenerationInput
from bilby_pipe.utils import convert_string_to_dict
from bilby_pipe.parser import create_parser
from bilby_pipe.main import parse_args
import sys
import bilby_pipe.utils
from ..utility import (log_version_information,
                       logger)
from ..calibration import build_calibration_lookup
from .. import prior as nullpol_prior
from .. import __version__
bilby_pipe.utils.logger = logger


class DataGenerationInput(BilbyDataGenerationInput):
    """Handles user-input for the data generation script

    Parameters
    ----------
    parser: configargparse.ArgParser, optional
        The parser containing the command line / ini file inputs
    args_list: list, optional
        A list of the arguments to parse. Defaults to `sys.argv[1:]`
    create_data: bool
        If false, no data is generated (used for testing)

    """
    def __init__(self, args, unknown_args, create_data=True):
        Input.__init__(self, args, unknown_args)

        # Generic initialisation
        self.meta_data = dict(
            command_line_args=args.__dict__,
            unknown_command_line_args=unknown_args,
            injection_parameters=None,
            nullpol_version=__version__,
        )
        self.injection_parameters = None

        # Admin arguments
        self.ini = args.ini
        self.transfer_files = args.transfer_files

        # Run index arguments
        self.idx = args.idx
        self.generation_seed = args.generation_seed
        self.trigger_time = args.trigger_time

        # Naming arguments
        self.outdir = args.outdir
        self.label = args.label

        # Prior arguments
        self.reference_frame = args.reference_frame
        self.time_reference = args.time_reference
        self.prior_file = args.prior_file
        self.prior_dict = args.prior_dict
        self.deltaT = args.deltaT
        self.default_prior = args.default_prior

        # Data arguments
        self.ignore_gwpy_data_quality_check = args.ignore_gwpy_data_quality_check
        self.detectors = args.detectors
        self.channel_dict = args.channel_dict
        self.data_dict = args.data_dict
        self.data_format = args.data_format
        self.allow_tape = args.allow_tape
        self.tukey_roll_off = args.tukey_roll_off
        self.gaussian_noise = args.gaussian_noise
        self.zero_noise = args.zero_noise
        self.resampling_method = args.resampling_method

        if args.timeslide_dict is not None:
            self.timeslide_dict = convert_string_to_dict(args.timeslide_dict)
            logger.info(f"Read-in timeslide dict directly: {self.timeslide_dict}")
        elif args.timeslide_file is not None:
            self.gps_file = args.gps_file
            self.timeslide_file = args.timeslide_file
            self.timeslide_dict = self.get_timeslide_dict(self.idx)

        # Data duration arguments
        self.duration = args.duration
        self.post_trigger_duration = args.post_trigger_duration

        # Frequencies
        self.sampling_frequency = args.sampling_frequency
        self.minimum_frequency = args.minimum_frequency
        self.maximum_frequency = args.maximum_frequency
        self.wavelet_frequency_resolution = args.wavelet_frequency_resolution
        self.wavelet_nx = args.wavelet_nx

        # Waveform, source model and likelihood
        self.waveform_generator_class = args.waveform_generator
        self.waveform_approximant = args.waveform_approximant
        self.catch_waveform_errors = args.catch_waveform_errors
        self.pn_spin_order = args.pn_spin_order
        self.pn_tidal_order = args.pn_tidal_order
        self.pn_phase_order = args.pn_phase_order
        self.pn_amplitude_order = args.pn_amplitude_order
        self.mode_array = args.mode_array
        self.waveform_arguments_dict = args.waveform_arguments_dict
        self.numerical_relativity_file = args.numerical_relativity_file
        self.injection_waveform_approximant = args.injection_waveform_approximant
        self.frequency_domain_source_model = args.frequency_domain_source_model
        self.conversion_function = args.conversion_function
        self.generation_function = args.generation_function
        self.likelihood_type = args.likelihood_type
        self.extra_likelihood_kwargs = args.extra_likelihood_kwargs
        self.enforce_signal_duration = args.enforce_signal_duration

        # PSD
        self.psd_maximum_duration = args.psd_maximum_duration
        self.psd_dict = args.psd_dict
        self.psd_length = args.psd_length
        self.psd_fractional_overlap = args.psd_fractional_overlap
        self.psd_start_time = args.psd_start_time
        self.psd_method = args.psd_method

        # Calibration
        self.calibration_model = args.calibration_model
        self.calibration_correction_type = args.calibration_correction_type
        self.spline_calibration_envelope_dict = args.spline_calibration_envelope_dict
        self.spline_calibration_amplitude_uncertainty_dict = (
            args.spline_calibration_amplitude_uncertainty_dict
        )
        self.spline_calibration_phase_uncertainty_dict = (
            args.spline_calibration_phase_uncertainty_dict
        )
        self.spline_calibration_nodes = args.spline_calibration_nodes
        self.calibration_prior_boundary = args.calibration_prior_boundary

        # Marginalization
        self.calibration_marginalization = args.calibration_marginalization
        self.calibration_lookup_table = args.calibration_lookup_table
        self.calibration_psd_lookup_table = args.calibration_psd_lookup_table
        self.number_of_response_curves = args.number_of_response_curves

        # Plotting
        self.plot_data = args.plot_data
        self.plot_spectrogram = args.plot_spectrogram
        self.plot_injection = args.plot_injection

        if create_data:
            self.create_data(args)

    @property
    def combined_default_prior_dicts(self):
        d = nullpol_prior.__dict__.copy()
        return d

    def build_calibration_lookups_if_needed(self):
        """
        Build lookup files that are needed for incorporating calibration uncertainty.
        These are needed if:

          - :code:`calibration_marginalization` is used either during sampling or
            post-processing
          - the calibration model is :code:`Precomputed`
        """
        sampling_calibration = self.calibration_model
        sampling_marginalization = self.calibration_marginalization
        calibration_lookup = self.calibration_lookup_table
        calibration_psd_lookup = self.calibration_psd_lookup_table
        n_response = self.number_of_response_curves
        if self.reweighting_configuration is not None:
            data = self.reweighting_configuration
            if "calibration-model" in data:
                self.calibration_model = data["calibration-model"]
                for ifo in self.interferometers:
                    self.add_calibration_model_to_interferometers(ifo)
            if "calibration-marginalization" in data:
                self.calibration_marginalization = data["calibration-marginalization"]
            if "calibration-lookup-table" in data:
                self.calibration_lookup_table = data["calibration-lookup-table"]
            if "calibration-psd-lookup-table" in data:
                self.calibration_psd_lookup_table = data["calibration-psd-lookup-table"]
            if "number-of-response-curves" in data:
                self.number_of_response_curves = data["number-of-response-curves"]

        if (
            self.calibration_marginalization
            or self.calibration_model == "Precomputed"
            or self.calibration_lookup_table
            or self.calibration_psd_lookup_table
        ):
            build_calibration_lookup(
                interferometers=self.interferometers,
                lookup_files=self.calibration_lookup_table,
                psd_lookup_files=self.calibration_psd_lookup_table,
                priors=self.calibration_prior,
                number_of_response_curves=self.number_of_response_curves,
            )
        self.calibration_model = sampling_calibration
        for ifo in self.interferometers:
            self.add_calibration_model_to_interferometers(ifo)
        self.calibration_marginalization = sampling_marginalization
        self.calibration_lookup_table = calibration_lookup
        self.calibration_psd_lookup_table = calibration_psd_lookup
        self.number_of_response_curves = n_response

def create_generation_parser():
    """Data generation parser creation"""
    return create_parser(top_level=False)

def main():
    """Data generation main logic"""
    args, unknown_args = parse_args(sys.argv[1:], create_generation_parser())
    log_version_information()
    data = DataGenerationInput(args, unknown_args)
    data.save_data_dump()
    logger.info("Completed data generation")