import bilby
from bilby_pipe.data_generation import DataGenerationInput as BilbyDataGenerationInput
from bilby_pipe.utils import convert_string_to_dict, DataDump
from bilby_pipe.main import parse_args
import sys
import bilby_pipe.utils
import numpy as np
import json
from .input import Input
from .parser import create_nullpol_parser
from ..utility import (logger,
                       get_file_extension,
                       is_file)
from ..calibration import build_calibration_lookup
from ..clustering import (run_time_frequency_clustering,
                          plot_spectrogram)
from ..null_stream import (encode_polarization,
                           relative_amplification_factor_map)
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
        self.time_frequency_clustering_time_padding = args.time_frequency_clustering_time_padding
        self.time_frequency_clustering_frequency_padding = args.time_frequency_clustering_frequency_padding
        self.time_frequency_clustering_skypoints = args.time_frequency_clustering_skypoints

        if create_data:
            self.create_data(args)

    @property
    def priors(self):
        """Read in and compose the prior at run-time"""
        if getattr(self, "_priors", None) is None:
            self._priors = self._get_priors()
            self._add_default_relative_polarization_prior()
        return self._priors

    def _add_default_relative_polarization_prior(self):
        # Encode the polarization modes
        polarization_modes, polarization_basis, polarization_derived = encode_polarization(self.polarization_modes,
                                                                                           self.polarization_basis)
        
        # Obtain the keywords
        relative_polarization_parameters = relative_amplification_factor_map(polarization_basis=polarization_basis,
                                                                             polarization_derived=polarization_derived).flatten()
        prior_remove_list = []
        for prior in self._priors:
            if prior[:4] == 'amp_' and prior[4:] not in relative_polarization_parameters:
                prior_remove_list.append(prior)
            elif prior[:6] == 'phase_' and prior[6:] not in relative_polarization_parameters:
                prior_remove_list.append(prior)

        # Remove the redundant parameters
        for i in range(len(prior_remove_list)):
            logger.debug(f'Removed irrelevant prior: {self._prior[prior_remove_list[i]]}')
            self._prior.pop(prior_remove_list[i])
            
        # Add missing prior
        missing_priors = {}
        for label in relative_polarization_parameters:
            name = f'amp_{label}'
            if name not in self._priors:
                missing_priors[name] = bilby.core.prior.Uniform(name=name,
                                                                minimum=0.0,
                                                                maximum=1.0)
                logger.debug(f'Added missing relative polarization prior: {missing_priors[name]}')
            name = f'phase_{label}'
            if name not in self._priors:
                missing_priors[name] = bilby.core.prior.Uniform(name=name,
                                                                minimum=0.0,
                                                                maximum=2. * np.pi,
                                                                boundary='periodic')
                logger.debug(f'Added missing relative polarization prior: {missing_priors[name]}')
        self._priors.update(missing_priors)
        
    @priors.setter
    def priors(self, priors):
        self._priors = priors

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

    def run_time_frequency_clustering(self):
        time_frequency_filter_file_provided = False
        logger.info(f"Running time-frequency clustering with method: {self.time_frequency_clustering_method}")
        # Build the strain data for time-frequency clustering
        if self.time_frequency_clustering_method == "data":
            frequency_domain_strain_array = np.array([ifo.frequency_domain_strain for ifo in self.interferometers])
        elif self.time_frequency_clustering_method in ["injection",
                                                       "injection_parameters_file",
                                                       "maxL",
                                                       "maP",
                                                       "random",
                                                       "time_frequency_filter_file"]:
            if self.time_frequency_clustering_method == "injection":
                parameters = self.meta_data["injection_parameters"]
            elif self.time_frequency_clustering_method == "injection_parameters_file":
                if self.time_frequency_clustering_injection_parameters_filename is None:
                    raise ValueError("time-frequency-clustering-injection-parameters-filename must be provided when time-frequency-clustering-method = 'injection_parameters_file'.")
                if get_file_extension(self.time_frequency_clustering_injection_parameters_filename) != ".json":
                    raise ValueError(f"time-frequency-clustering-injection-parameters-filename = {self.time_frequency_clustering_injection_parameters_filename} needs to be a JSON file.")
                with open(self.time_frequency_clustering_injection_parameters_filename, 'r') as f:
                    parameters = json.load(f)
                if isinstance(parameters, list):
                    if len(parameters) > 1:
                        logger.warning(f"Currently only supports injecting a single signal, but the injection parameters file contains {len(parameters)} signals.")
                        logger.warning(f"Only the first signal is injected.")
                    parameters = parameters[0]
                elif isinstance(parameters, dict):
                    if "injections" in parameters:
                        parameters = parameters['injections']
                        if isinstance(parameters, list):
                            if len(parameters) > 1:
                                logger.warning(f"Currently only supports injecting a single signal, but the injection parameters file contains {len(parameters)} signals.")
                                logger.warning(f"Only the first signal is injected.")
                            parameters = parameters[0]
                else:
                    raise ValueError(f"Data type of time-frequency-clustering-injection-parameters-filename = {type(parameters)} is not supported. Expected a list or a dictionary.")
            else:
                if self.time_frequency_clustering_pe_samples_filename is None:
                    raise ValueError("PE samples filename must be provided for PE-samples-based time-frequency clustering")
                posterior = bilby.core.result.read_in_result(self.time_frequency_clustering_pe_samples_filename).posterior
                if self.time_frequency_clustering_method == "maxL":
                    parameters = posterior.loc[posterior["log_likelihood"].idxmax()].to_dict()
                elif self.time_frequency_clustering_method == "maP":
                    parameters = posterior.loc[(posterior["log_likelihood"]+posterior["log_prior"]).idxmax()].to_dict()
                elif self.time_frequency_clustering_method == "random":
                    parameters = posterior.sample().to_dict()
                else:
                    raise ValueError(f"Unknown time-frequency clustering method {self.time_frequency_clustering_method}")
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
                parameters=self.injection_parameters,
                raise_error=self.enforce_signal_duration,
            )
            frequency_domain_strain_array = np.array([ifo.frequency_domain_strain for ifo in self.interferometers])
        elif is_file(self.time_frequency_clustering_method):
            frequency_domain_strain_array = np.array([ifo.frequency_domain_strain for ifo in self.interferometers])
            time_frequency_filter_file_provided = True
        else:
            raise ValueError(
                f"Unknown time-frequency clustering method {self.time_frequency_clustering_method}"
            )
        time_frequency_filter, sky_maximized_spectrogram = run_time_frequency_clustering(interferometers=self.interferometers,
                                                                                         frequency_domain_strain_array=frequency_domain_strain_array,
                                                                                         wavelet_frequency_resolution=self.wavelet_frequency_resolution,
                                                                                         wavelet_nx=self.wavelet_nx,
                                                                                         minimum_frequency=self.minimum_frequency,
                                                                                         maximum_frequency=self.maximum_frequency,
                                                                                         threshold=self.time_frequency_clustering_threshold,
                                                                                         time_padding=self.time_frequency_clustering_time_padding,
                                                                                         frequency_padding=self.time_frequency_clustering_frequency_padding,
                                                                                         skypoints=self.time_frequency_clustering_skypoints,
                                                                                         return_sky_maximized_spectrogram=True)
        if time_frequency_filter_file_provided:
            time_frequency_filter = np.load(self.time_frequency_clustering_method)
            logger.info(f'Loaded time-frequency filter from {self.time_frequency_clustering_method}.')
        self.meta_data['time_frequency_filter'] = time_frequency_filter
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
    data.run_time_frequency_clustering()
    data.save_data_dump()
    logger.info("Completed data generation")
