import bilby
from bilby_pipe.data_generation import DataGenerationInput as BilbyDataGenerationInput
from bilby_pipe.utils import convert_string_to_dict, DataDump
from bilby_pipe.main import parse_args
import sys
import bilby_pipe.utils
import numpy as np
from .input import Input
from .parser import create_nullpol_parser
from ..psd import simulate_psd_from_bilby_psd
from ..utility import (logger,
                       NullpolError,
                       is_file)
from ..calibration import build_calibration_lookup
from ..clustering import (run_time_frequency_clustering as _run_time_frequency_clustering,
                          compute_sky_maximized_spectrogram,
                          plot_spectrogram,
                          compute_logistic_probability_map)
from .. import (__version__,
                log_version_information)
bilby_pipe.utils.logger = logger


class DataGenerationInput(BilbyDataGenerationInput, Input):
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

        # PSD
        self.psd_maximum_duration = args.psd_maximum_duration
        self.psd_dict = args.psd_dict
        self.psd_length = args.psd_length
        self.psd_fractional_overlap = args.psd_fractional_overlap
        self.psd_start_time = args.psd_start_time
        self.psd_method = args.psd_method
        self.simulate_psd_nsample = args.simulate_psd_nsample

        # Calibration
        self.calibration_model = args.calibration_model
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

        # Time-frequency clustering
        self.time_frequency_clustering_method = args.time_frequency_clustering_method
        self.time_frequency_clustering_injection_parameters_filename = args.time_frequency_clustering_injection_parameters_filename
        self.time_frequency_clustering_pe_samples_filename = args.time_frequency_clustering_pe_samples_filename
        self.time_frequency_clustering_threshold = args.time_frequency_clustering_threshold
        self.time_frequency_clustering_threshold_type = args.time_frequency_clustering_threshold_type
        self.time_frequency_clustering_time_padding = args.time_frequency_clustering_time_padding
        self.time_frequency_clustering_frequency_padding = args.time_frequency_clustering_frequency_padding
        self.time_frequency_clustering_skypoints = args.time_frequency_clustering_skypoints
        self.time_frequency_probability_map_confidence_threshold = args.time_frequency_probability_map_confidence_threshold
        self.time_frequency_probability_map_steepness = args.time_frequency_probability_map_steepness

        if create_data:
            self.create_data(args)

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

    @property
    def time_frequency_clustering_method(self):
        return getattr(self, "_time_frequency_clustering_method", None)

    @time_frequency_clustering_method.setter
    def time_frequency_clustering_method(self, method):
        self._time_frequency_clustering_method = method

    @property
    def time_frequency_clustering_pe_samples_filename(self):
        return getattr(self, "_time_frequency_clustering_pe_samples_filename", None)

    @time_frequency_clustering_pe_samples_filename.setter
    def time_frequency_clustering_pe_samples_filename(self, filename):
        self._time_frequency_clustering_pe_samples_filename = filename

    @property
    def time_frequency_clustering_threshold(self):
        return getattr(self, "_time_frequency_clustering_threshold", None)

    @time_frequency_clustering_threshold.setter
    def time_frequency_clustering_threshold(self, threshold):
        self._time_frequency_clustering_threshold = threshold

    @property
    def time_frequency_clustering_threshold_type(self):
        return getattr(self, "_time_frequency_clustering_threshold_type", None)

    @time_frequency_clustering_threshold_type.setter
    def time_frequency_clustering_threshold_type(self, threshold_type):
        self._time_frequency_clustering_threshold_type = threshold_type

    @property
    def time_frequency_clustering_time_padding(self):
        return getattr(self, "_time_frequency_clustering_time_padding", None)

    @time_frequency_clustering_time_padding.setter
    def time_frequency_clustering_time_padding(self, time_padding):
        self._time_frequency_clustering_time_padding = time_padding

    @property
    def time_frequency_clustering_frequency_padding(self):
        return getattr(self, "_time_frequency_clustering_frequency_padding", None)

    @time_frequency_clustering_frequency_padding.setter
    def time_frequency_clustering_frequency_padding(self, frequency_padding):
        self._time_frequency_clustering_frequency_padding = frequency_padding

    @property
    def time_frequency_clustering_skypoints(self):
        return getattr(self, "_time_frequency_clustering_skypoints", None)

    @time_frequency_clustering_skypoints.setter
    def time_frequency_clustering_skypoints(self, skypoints):
        self._time_frequency_clustering_skypoints = skypoints

    @property
    def time_frequency_probability_map_confidence_threshold(self):
        return getattr(self, "_time_frequency_probability_map_confidence_threshold", None)

    @time_frequency_probability_map_confidence_threshold.setter
    def time_frequency_probability_map_confidence_threshold(self, threshold):
        self._time_frequency_probability_map_confidence_threshold = threshold

    @property
    def time_frequency_probability_map_steepness(self):
        return getattr(self, "_time_frequency_probability_map_steepness", None)

    @time_frequency_probability_map_steepness.setter
    def time_frequency_probability_map_steepness(self, steepness):
        self._time_frequency_probability_map_steepness = steepness

    def _get_interferometers_from_injection_in_gaussian_noise(self):
        # Copy the interferometers
        interferometers = bilby.gw.detector.InterferometerList([ifo.name for ifo in self.interferometers])
        for i in range(len(interferometers)):            
            power_spectral_density = bilby.gw.detector.PowerSpectralDensity(frequency_array=self.interferometers[i].frequency_array.copy(),
                                                                            psd_array=self.interferometers[i].power_spectral_density_array.copy())
            interferometers[i].power_spectral_density = power_spectral_density
        injection_parameters = self.injection_df.iloc[self.idx].to_dict()
        # Set the strain data from zero noise.
        interferometers.set_strain_data_from_zero_noise(sampling_frequency=self.sampling_frequency,
                                                        duration=self.duration,
                                                        start_time=self.start_time)
        waveform_arguments = self.get_injection_waveform_arguments()
        waveform_generator = self.waveform_generator_class(duration=self.duration,
                                                           start_time=self.start_time,
                                                           sampling_frequency=self.sampling_frequency,
                                                           frequency_domain_source_model=self.bilby_frequency_domain_source_model,
                                                           parameter_conversion=self.parameter_conversion,
                                                           waveform_arguments=waveform_arguments)
        interferometers.inject_signal(waveform_generator=waveform_generator,
                                      parameters=injection_parameters,
                                      raise_error=self.enforce_signal_duration)
        return interferometers

    def estimate_wavelet_psd(self):
        # Estimate the wavelet PSD
        logger.info('Estimating wavelet PSDs...')
        psd_array = np.array([simulate_psd_from_bilby_psd(psd=ifo.power_spectral_density,
                                                            seglen=ifo.duration,
                                                            srate=ifo.sampling_frequency,
                                                            wavelet_frequency_resolution=self.wavelet_frequency_resolution,
                                                            nsample=self.simulate_psd_nsample,
                                                            nx=self.wavelet_nx) for ifo in self.interferometers])
        # Save the wavelet PSD to disk.
        self.meta_data['wavelet_psd_array'] = psd_array

    def run_time_frequency_clustering(self):
        logger.info(f"Running time-frequency clustering with method: {self.time_frequency_clustering_method}")
        # Build the strain data for time-frequency clustering
        if self.time_frequency_clustering_method in ["data",
                                                     "injection",
                                                     "injection_parameters_file",
                                                     "maxL",
                                                     "maP",
                                                     "random"]:        
            if self.time_frequency_clustering_method == "data":
                frequency_domain_strain_array = np.array([ifo.frequency_domain_strain for ifo in self.interferometers])
            elif self.time_frequency_clustering_method in ["injection",
                                                           "injection_parameters_file",
                                                           "maxL",
                                                           "maP",
                                                           "random"]:
                if self.time_frequency_clustering_method == "injection":
                    parameters = self.meta_data["injection_parameters"]
                elif self.time_frequency_clustering_method == "injection_parameters_file":
                    if self.time_frequency_clustering_injection_parameters_filename is None:
                        raise NullpolError("time-frequency-clustering-injection-parameters-filename must be provided when time-frequency-clustering-method = 'injection_parameters_file'.")
                    injection_df = Input.read_injection_file(self.time_frequency_clustering_injection_parameters_filename)
                    if len(injection_df) > 1:
                        logger.warning(f"More than one injections in time-frequency-clustering-injection-parameters-filename={self.time_frequency_clustering_injection_parameters_filename}.")
                        logger.warning("Use the first injection.")
                    parameters = injection_df.iloc[0].to_dict()
                elif self.time_frequency_clustering_method in ["maxL",
                                                               "maP",
                                                               "random"]:
                    if self.time_frequency_clustering_pe_samples_filename is None:
                        raise NullpolError(f"time-frequency-clustering-method = {self.time_frequency_clustering_method}, but time-frequency-clustering-pe-samples-filename is not provided.")
                    posterior = bilby.core.result.read_in_result(self.time_frequency_clustering_pe_samples_filename).posterior
                    if self.time_frequency_clustering_method == "maxL":
                        parameters = posterior.loc[posterior["log_likelihood"].idxmax()].to_dict()
                    elif self.time_frequency_clustering_method == "maP":
                        parameters = posterior.loc[(posterior["log_likelihood"]+posterior["log_prior"]).idxmax()].to_dict()
                    elif self.time_frequency_clustering_method == "random":
                        parameters = posterior.sample().to_dict()
                    else:
                        raise NullpolError(f"Unexpected error with time-frequency-clustering-method = {self.time_frequency_clustering_method}. Contact the developers.")
                    # Remove log_likelihood and log_prior from parameters
                    parameters.pop("log_likelihood", None)
                    parameters.pop("log_prior", None)
                # Generate the mock strain data from the parameters.
                ## Generate a new interferometer list with the same detectors
                logger.info("Generating zero-noise injection data")
                ifos = bilby.gw.detector.InterferometerList([ifo.name for ifo in self.interferometers])
                # Copy the power spectral density
                for i in range(len(ifos)):
                    ifos[i].power_spectral_density = self.interferometers[i].power_spectral_density
                ifos.set_strain_data_from_zero_noise(
                    sampling_frequency=self.sampling_frequency,
                    duration=self.duration,
                    start_time=self.start_time
                )
                waveform_arguments = self.get_injection_waveform_arguments()
                logger.info(f"Using waveform arguments: {waveform_arguments}")
                waveform_generator = self.waveform_generator_class(
                    duration=self.duration,
                    start_time=self.start_time,
                    sampling_frequency=self.sampling_frequency,
                    frequency_domain_source_model=self.bilby_frequency_domain_source_model,
                    parameter_conversion=self.parameter_conversion,
                    waveform_arguments=waveform_arguments,
                )
                ifos.inject_signal(
                    waveform_generator=waveform_generator,
                    parameters=parameters,
                    raise_error=self.enforce_signal_duration,
                )
                frequency_domain_strain_array = np.array([ifo.frequency_domain_strain for ifo in ifos])
            else:
                raise ValueError(
                    f"Unknown time-frequency clustering method {self.time_frequency_clustering_method}"
                )
            time_frequency_filter, sky_maximized_spectrogram = _run_time_frequency_clustering(interferometers=self.interferometers,
                                                                                             frequency_domain_strain_array=frequency_domain_strain_array,
                                                                                             wavelet_frequency_resolution=self.wavelet_frequency_resolution,
                                                                                             wavelet_nx=self.wavelet_nx,
                                                                                             minimum_frequency=self.minimum_frequency,
                                                                                             maximum_frequency=self.maximum_frequency,
                                                                                             threshold=self.time_frequency_clustering_threshold,
                                                                                             time_padding=self.time_frequency_clustering_time_padding,
                                                                                             frequency_padding=self.time_frequency_clustering_frequency_padding,
                                                                                             skypoints=self.time_frequency_clustering_skypoints,
                                                                                             return_sky_maximized_spectrogram=True,
                                                                                             psd_array=self.meta_data['wavelet_psd_array'],
                                                                                             threshold_type=self.time_frequency_clustering_threshold_type)
        elif is_file(self.time_frequency_clustering_method):
            time_frequency_filter = np.load(self.time_frequency_clustering_method)
            sky_maximized_spectrogram = None
            logger.info(f'Loaded time-frequency filter from {self.time_frequency_clustering_method}.')
        else:
            raise NullpolError(f"Unrecognized time-frequency-clustering-method = {self.time_frequency_clustering_method}.")
        # Generate the sky-maximized spectrogram from data
        sky_maximized_spectrogram_data = compute_sky_maximized_spectrogram(interferometers=self.interferometers,
                                                                           frequency_domain_strain_array=np.array([ifo.frequency_domain_strain for ifo in self.interferometers]),
                                                                           wavelet_frequency_resolution=self.wavelet_frequency_resolution,
                                                                           wavelet_nx=self.wavelet_nx,
                                                                           minimum_frequency=self.minimum_frequency,
                                                                           maximum_frequency=self.maximum_frequency,
                                                                           skypoints=self.time_frequency_clustering_skypoints,
                                                                           psd_array=self.meta_data['wavelet_psd_array'])
        if self.likelihood_type == 'WeightedGaussianTimeFrequencyLikelihood':
            logger.info('Computing logistic probability map...')
            time_frequency_filter = compute_logistic_probability_map(whitened_time_frequency_spectrogram=sky_maximized_spectrogram_data,
                                                                     confidence_threshold=self.time_frequency_probability_map_confidence_threshold,
                                                                     df=len(self.interferometers),
                                                                     time_frequency_filter=time_frequency_filter,
                                                                     steepness=self.time_frequency_probability_map_steepness)
        self.meta_data['time_frequency_filter'] = time_frequency_filter
        self.meta_data['sky_maximized_spectrogram'] = sky_maximized_spectrogram
        self.meta_data['sky_maximized_spectrogram_data'] = sky_maximized_spectrogram_data
        # Plot
        time_frequency_filter_fig_fname = f"{self.data_directory}/{self.label}_time_frequency_filter_spectrogram.png"
        plot_spectrogram(spectrogram=time_frequency_filter,
                         duration=self.interferometers[0].duration,
                         sampling_frequency=self.interferometers[0].sampling_frequency,
                         wavelet_frequency_resolution=self.wavelet_frequency_resolution,
                         t0=self.start_time,
                         title="Time-frequency Filter Spectrogram",
                         savefig=time_frequency_filter_fig_fname,
                         dpi=100)
        logger.info(f"Saved plot of time-frequency filter spectrogram to {time_frequency_filter_fig_fname}.")
        # Plot
        if sky_maximized_spectrogram is not None:
            spectrogram_fig_fname = f"{self.data_directory}/{self.label}_sky_maximized_spectrogram.png"
            plot_spectrogram(spectrogram=sky_maximized_spectrogram,
                            duration=self.interferometers[0].duration,
                            sampling_frequency=self.interferometers[0].sampling_frequency,
                            wavelet_frequency_resolution=self.wavelet_frequency_resolution,
                            t0=self.start_time,
                            title="Sky-maximized Spectrogram",
                            savefig=spectrogram_fig_fname,
                            dpi=100)
            logger.info(f"Saved plot of sky-maximized spectrogram to {spectrogram_fig_fname}.")
        # Plot
        spectrogram_data_fig_fname = f"{self.data_directory}/{self.label}_sky_maximized_spectrogram_data.png"
        plot_spectrogram(spectrogram=sky_maximized_spectrogram_data,
                         duration=self.interferometers[0].duration,
                         sampling_frequency=self.interferometers[0].sampling_frequency,
                         wavelet_frequency_resolution=self.wavelet_frequency_resolution,
                         t0=self.start_time,
                         title="Sky-maximized Spectrogram Data",
                         savefig=spectrogram_data_fig_fname,
                         dpi=100)
        logger.info(f"Saved plot of sky-maximized spectrogram of data to {spectrogram_data_fig_fname}.")

    def save_data_dump(self):
        """Method to dump the saved data to disk for later analysis"""
        self.build_calibration_lookups_if_needed()
        self.meta_data["reweighting_configuration"] = self.reweighting_configuration
        data_dump = DataDump(
            outdir=self.data_directory,
            label=self.label,
            idx=self.idx,
            trigger_time=self.trigger_time,
            interferometers=self.interferometers,
            meta_data=self.meta_data,
            likelihood_lookup_table=None,
            likelihood_roq_weights=None,
            likelihood_roq_params=None,
            likelihood_multiband_weights=None,
            priors_dict=dict(self.priors),
            priors_class=self.priors.__class__,
        )
        data_dump.to_pickle()

def create_generation_parser():
    """Data generation parser creation"""
    return create_nullpol_parser(top_level=False)

def main():
    """Data generation main logic"""
    args, unknown_args = parse_args(sys.argv[1:], create_generation_parser())
    log_version_information()
    data = DataGenerationInput(args, unknown_args)
    data.estimate_wavelet_psd()
    data.run_time_frequency_clustering()
    data.save_data_dump()
    logger.info("Completed data generation")
