import bilby
from bilby_pipe.data_analysis import DataAnalysisInput as BilbyDataAnalysisInput
from bilby_pipe.main import parse_args
from bilby_pipe.utils import (
    CHECKPOINT_EXIT_CODE,
    BilbyPipeError,
    DataDump,
    resolve_filename_with_transfer_fallback,
    convert_string_to_dict,
)
import bilby_pipe.utils
import signal
import sys
import numpy as np
from .parser import create_nullpol_parser
from .input import Input
from ..utility import (logger,
                       log_version_information)
from ..result import PolarizationResult
from ..clustering import (run_time_frequency_clustering,
                          write_time_frequency_filter)

# fmt: off
import matplotlib  # isort:skip
matplotlib.use("agg")
# fmt: on
bilby_pipe.utils.logger = logger


def sighandler(signum, frame):
    logger.info("Performing periodic eviction")
    sys.exit(CHECKPOINT_EXIT_CODE)

class DataAnalysisInput(BilbyDataAnalysisInput, Input):
    """Handles user-input for the data analysis script.
    """
    def __init__(self, args, unknown_args, test=False):
        """Initializer.

        Args:
            args (tuple): A tuple of arguments.
            unknown_args (tuple): A tuple of unknown arguments.
            test (bool, optional): _description_. Defaults to False.
        """
        Input.__init__(self, args, unknown_args)

        # Generic initialisation
        self.meta_data = dict()
        self.result = None

        # Admin arguments
        self.ini = args.ini
        self.scheduler = args.scheduler
        self.periodic_restart_time = args.periodic_restart_time
        self.request_cpus = args.request_cpus
        self.run_local = args.local

        # Data dump file to run on
        self.data_dump_file = args.data_dump_file

        # Choices for running
        self.detectors = args.detectors
        self.sampler = args.sampler
        self.sampler_kwargs = args.sampler_kwargs
        self.sampling_seed = args.sampling_seed

        # Frequencies
        self.sampling_frequency = args.sampling_frequency
        self.minimum_frequency = args.minimum_frequency
        self.maximum_frequency = args.maximum_frequency
        self.wavelet_frequency_resolution = args.wavelet_frequency_resolution
        self.wavelet_nx = args.wavelet_nx
        self.simulate_psd_nsample = args.simulate_psd_nsample

        # Time-frequency clustering
        self.time_frequency_clustering_method = args.time_frequency_clustering_method
        self.time_frequency_clustering_pe_samples_filename = args.time_frequency_clustering_pe_samples_filename
        self.time_frequency_clustering_threshold = args.time_frequency_clustering_threshold
        self.time_frequency_clustering_time_padding = args.time_frequency_clustering_time_padding
        self.time_frequency_clustering_skypoints = args.time_frequency_clustering_skypoints

        # Likelihood
        self.likelihood_type = args.likelihood_type
        self.polarization_modes = args.polarization_modes
        self.polarization_basis = args.polarization_basis
        self.extra_likelihood_kwargs = args.extra_likelihood_kwargs

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
        self.number_of_response_curves = args.number_of_response_curves

        if test is False:
            self._load_data_dump()

    @property
    def result_class(self):
        """The nullpol result class to store results in"""
        return PolarizationResult
    
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
    def time_frequency_clustering_time_skypoints(self, skypoints):
        self._time_frequency_clustering_skypoints = skypoints

    def get_likelihood_and_priors(self):
        """Read in the likelihood and prior from the data dump

        This reads in the data dump values and reconstructs the likelihood and
        priors. Note, care must be taken to use the "search_priors" which differ
        from the true prior when using marginalization

        Returns
        -------
        likelihood, priors
            The bilby likelihood and priors
        """

        priors = self.data_dump.priors_class(self.data_dump.priors_dict)
        self.priors = priors
        likelihood = self.likelihood
        priors = self.search_priors
        return likelihood, priors
        
    def run_time_frequency_clustering(self):
        if self.scheduler.lower() == "condor" and not self.run_local:
            signal.signal(signal.SIGALRM, handler=sighandler)
            signal.alarm(self.periodic_restart_time)
        logger.info(f"Running time-frequency clustering with method: {self.time_frequency_clustering_method}")
        # Build the strain data for time-frequency clustering
        if self.time_frequency_clustering_method == "data":
            frequency_domain_strain_array = np.array([ifo.frequency_domain_strain for ifo in self.interferometers])
        elif self.time_frequency_clustering_method in ["injection",
                                                       "maxL",
                                                       "maP",
                                                       "random"]:
            if self.time_frequency_clustering_method == "injection":
                parameters = self.meta_data["injection_parameters"]
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
                parameters.pop("log_likelihood")
                parameters.pop("log_prior")
            # Generate the mock strain data from the parameters.
            ## Generate a new interferometer list with the same detectors
            logger.info("Generating zero-noise injection data")
            ifos = bilby.gw.detector.InterferometerList([ifo.name for ifo in self.interferometers])
            # Copy the power spectral density
            for ifo in ifos:
                ifo.power_spectral_density = self.interferometers[ifo.name].power_spectral_density
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
                frequency_domain_source_model=self.injection_bilby_frequency_domain_source_model,
                parameter_conversion=self.parameter_conversion,
                waveform_arguments=waveform_arguments,
            )
            ifos.inject_signal(
                waveform_generator=waveform_generator,
                parameters=self.injection_parameters,
                raise_error=self.enforce_signal_duration,
            )
            frequency_domain_strain_array = np.array([ifo.frequency_domain_strain for ifo in self.interferometers])
        else:
            raise ValueError(
                f"Unknown time-frequency clustering method {self.time_frequency_clustering_method}"
            )            
        time_frequency_filter = run_time_frequency_clustering(interferometers=self.interferometers,
                                                              frequency_domain_strain_array=frequency_domain_strain_array,
                                                              wavelet_frequency_resolution=self.wavelet_frequency_resolution,
                                                              wavelet_nx=self.wavelet_nx,
                                                              threshold=self.time_frequency_clustering_threshold,
                                                              time_padding=self.time_frequency_clustering_time_padding,
                                                              frequency_padding=self.time_frequency_clustering_frequency_padding,
                                                              skypoints=self.time_frequency_clustering_skypoints)
        write_time_frequency_filter(f"{self.label}_time_frequency_filter.npy", time_frequency_filter)

    def run_sampler(self):
        if self.scheduler.lower() == "condor" and not self.run_local:
            signal.signal(signal.SIGALRM, handler=sighandler)
            signal.alarm(self.periodic_restart_time)

        likelihood, priors = self.get_likelihood_and_priors()

        self.result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler=self.sampler,
            label=self.label,
            outdir=self.result_directory,
            conversion_function=None,
            injection_parameters=self.meta_data["injection_parameters"],
            meta_data=self.meta_data,
            result_class=self.result_class,
            exit_code=CHECKPOINT_EXIT_CODE,
            save=self.result_format,
            **self.sampler_kwargs,
        )

def create_analysis_parser(usage=__doc__):
    """Data analysis parser creation"""
    return create_nullpol_parser(top_level=False)

def main():
    """Data analysis main logic"""
    args, unknown_args = parse_args(
        sys.argv[1:],
        create_analysis_parser(usage=__doc__),
    )
    log_version_information()
    analysis = DataAnalysisInput(args, unknown_args)
    analysis.run_time_frequency_clustering()
    analysis.run_sampler()
    if analysis.reweighting_configuration is not None:
        analysis.reweight_result()
    logger.info("Run completed")