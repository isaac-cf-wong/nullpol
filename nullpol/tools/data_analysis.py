import bilby
from bilby_pipe.main import parse_args
from bilby_pipe.parser import create_parser
from bilby_pipe.utils import (
    CHECKPOINT_EXIT_CODE,
    BilbyPipeError,
    DataDump,
    resolve_filename_with_transfer_fallback,
    convert_string_to_dict,
)
import bilby_pipe.utils
import os
import signal
import sys
import numpy as np
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

class DataAnalysisInput(Input):
    """Handles user-input for the data analysis script.
    """
    def __init__(self, args, unknown_args, test=False):
        """Initializer.

        Args:
            args (tuple): A tuple of arguments.
            unknown_args (tuple): A tuple of unknown arguments.
            test (bool, optional): _description_. Defaults to False.
        """
        super().__init__(args, unknown_args)

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
    def polarization_modes(self):
        return self._polarization_modes
    
    @polarization_modes.setter
    def polarization_modes(self, modes):
        self._polarization_modes = modes
        if modes is not None:
            logger.debug(f"Polarization modes set to {modes}")
        else:
            self._polarization_modes = 'pc'
            logger.debug(f"Polarization modes set to default value of {self._polarization_modes}")

    @property
    def polarization_basis(self):
        return self._polarization_basis
    
    @polarization_basis.setter
    def polarization_basis(self, basis):
        self._polarization_basis = basis
        if basis is not None:
            logger.debug(f"Polarization basis set to {basis}")
        else:
            self._polarization_basis = self.polarization_modes
            logger.debug(f"Polarization basis set to default value of {self._polarization_basis}")

    @property
    def sampling_seed(self):
        return self._sampling_seed

    @sampling_seed.setter
    def sampling_seed(self, sampling_seed):
        if sampling_seed is None:
            sampling_seed = np.random.randint(1, 1e6)
        self._sampling_seed = sampling_seed
        np.random.seed(sampling_seed)
        bilby.core.utils.random.seed(sampling_seed)
        logger.info(f"Sampling seed set to {sampling_seed}")

        if not any(
            [
                k in self.sampler_kwargs
                for k in bilby.core.sampler.Sampler.sampling_seed_equiv_kwargs
            ]
        ):
            self.sampler_kwargs["sampling_seed"] = self._sampling_seed

    @property
    def interferometers(self):
        try:
            return self._interferometers
        except AttributeError:
            ifos = self.data_dump.interferometers
            names = [ifo.name for ifo in ifos]
            logger.info(f"Found data for detectors = {names}")
            ifos_to_use = [ifo for ifo in ifos if ifo.name in self.detectors]
            names_to_use = [ifo.name for ifo in ifos_to_use]
            logger.info(f"Using data for detectors = {names_to_use}")
            self._interferometers = bilby.gw.detector.InterferometerList(ifos_to_use)
            self.print_detector_information(self._interferometers)
            return self._interferometers

    @staticmethod
    def print_detector_information(interferometers):
        for ifo in interferometers:
            logger.info(
                "{}: sampling-frequency={}, segment-start-time={}, duration={}".format(
                    ifo.name,
                    ifo.strain_data.sampling_frequency,
                    ifo.strain_data.start_time,
                    ifo.strain_data.duration,
                )
            )

    @property
    def data_dump(self):
        if hasattr(self, "_data_dump"):
            return self._data_dump
        else:
            raise BilbyPipeError("Data dump not loaded")

    def _load_data_dump(self):
        filename = self.data_dump_file
        self.meta_data["data_dump"] = filename

        logger.debug("Data dump not previously loaded")

        final_filename = resolve_filename_with_transfer_fallback(filename)
        if filename is None:
            raise FileNotFoundError(
                f"No dump data {filename} file found. Most likely the generation "
                "step failed."
            )

        self._data_dump = DataDump.from_pickle(final_filename)
        self.meta_data.update(self._data_dump.meta_data)
        return self._data_dump

    @property
    def result_class(self):
        """The nullpol result class to store results in"""
        return PolarizationResult
    
    @property
    def calibration_psd_lookup_table(self):
        return getattr(self, "_calibration_psd_lookup_table", None)

    @calibration_psd_lookup_table.setter
    def calibration_psd_lookup_table(self, lookup):
        if isinstance(lookup, str):
            lookup = convert_string_to_dict(lookup)
        self._calibration_psd_lookup_table = lookup

    @property
    def result_directory(self):
        result_dir = os.path.join(self.outdir, "result")
        return os.path.relpath(result_dir)

    @property
    def wavelet_nx(self):
        return self._wavelet_nx

    @wavelet_nx.setter
    def wavelet_nx(self, wavelet_nx):
        self._wavelet_nx = wavelet_nx
        if wavelet_nx is not None:
            logger.debug(f"wavelet_nx set to {wavelet_nx}")
        else:
            self._wavelet_nx = 4.
            logger.debug(f"wavelet_nx set to default value of 4.")

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

    @property
    def time_frequency_clustering_method(self):
        return self._time_frequency_clustering_method
    
    @time_frequency_clustering_method.setter
    def time_frequency_clustering_method(self, method):
        self._time_frequency_clustering_method = method
        if method is not None:
            logger.debug(f"Time-frequency clustering method set to {method}")
        else:
            self._time_frequency_clustering_method = "data"
            logger.debug(f"Time-frequency clustering method set to default value of '{self._time_frequency_clustering_method}'")

    @property
    def time_frequency_clustering_pe_samples_filename(self):
        return self._time_frequency_clustering_pe_samples_filename

    @time_frequency_clustering_pe_samples_filename.setter
    def time_frequency_clustering_pe_samples_filename(self, filename):
        self._time_frequency_clustering_pe_samples_filename = filename
        if filename is not None:
            logger.debug(f"Time-frequency clustering PE samples filename set to {filename}")
        
    @property
    def time_frequency_clustering_threshold(self):
        return self._time_frequency_clustering_threshold
    
    @time_frequency_clustering_threshold.setter
    def time_frequency_clustering_threshold(self, threshold):
        self._time_frequency_clustering_threshold = threshold
        if threshold is not None:
            logger.debug(f"Time-frequency clustering threshold set to {threshold}")
        else:
            self._time_frequency_clustering_threshold = 0.95
            logger.debug(f"Time-frequency clustering threshold set to default value of {self._time_frequency_clustering_threshold}")

    @property
    def time_frequency_clustering_time_padding(self):
        return self._time_frequency_clustering_time_padding
    
    @time_frequency_clustering_time_padding.setter
    def time_frequency_clustering_time_padding(self, time_padding):
        self._time_frequency_clustering_time_padding = time_padding
        if time_padding is not None:
            logger.debug(f"Time-frequency clustering time padding set to {time_padding}")
        else:
            self._time_frequency_clustering_time_padding = 0.1
            logger.debug(f"Time-frequency clustering time padding set to default value of {self._time_frequency_clustering_time_padding}")

    @property
    def time_frequency_clustering_skypoints(self):
        return self._time_frequency_clustering_skypoints
    
    @time_frequency_clustering_skypoints.setter
    def time_frequency_clustering_skypoints(self, skypoints):
        self._time_frequency_clustering_skypoints = skypoints
        if skypoints is not None:
            logger.debug(f"Time-frequency clustering skypoints set to {skypoints}")
        else:
            self._time_frequency_clustering_skypoints = 100
            logger.debug(f"Time-frequency clustering skypoints set to default value of {self._time_frequency_clustering_skypoints}")

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
    return create_parser(top_level=False, usage=usage)

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