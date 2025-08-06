from __future__ import annotations

import sys
from argparse import Namespace

import bilby
import bilby_pipe.utils
import h5py
import numpy as np
import pandas as pd
from bilby_pipe.data_generation import DataGenerationInput as BilbyDataGenerationInput
from bilby_pipe.main import parse_args
from bilby_pipe.utils import DataDump, convert_string_to_dict

from .. import __version__, log_version_information
from ..clustering import compute_sky_maximized_spectrogram, plot_reverse_cumulative_distribution, plot_spectrogram
from ..clustering import run_time_frequency_clustering as _run_time_frequency_clustering
from ..utils import NullpolError, is_file, logger
from .input import Input
from .parser import create_nullpol_parser

bilby_pipe.utils.logger = logger


class DataGenerationInput(BilbyDataGenerationInput, Input):
    """Handles user-input for the data generation script.

    Args:
        args (argparse.Namespace): Parsed command line arguments containing configuration.
        unknown_args (list): List of unrecognized command line arguments.
        create_data (bool, optional): If False, no data is generated (used for testing). Defaults to True.
    """

    def __init__(self, args: Namespace, unknown_args: list, create_data: bool = True):
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

        # Calibration
        self.calibration_model = args.calibration_model
        self.spline_calibration_envelope_dict = args.spline_calibration_envelope_dict
        self.spline_calibration_amplitude_uncertainty_dict = args.spline_calibration_amplitude_uncertainty_dict
        self.spline_calibration_phase_uncertainty_dict = args.spline_calibration_phase_uncertainty_dict
        self.spline_calibration_nodes = args.spline_calibration_nodes
        self.calibration_prior_boundary = args.calibration_prior_boundary

        # Plotting
        self.plot_data = args.plot_data
        self.plot_spectrogram = args.plot_spectrogram
        self.plot_injection = args.plot_injection

        # Time-frequency clustering
        self.time_frequency_clustering_method = args.time_frequency_clustering_method
        self.time_frequency_clustering_injection_parameters_filename = (
            args.time_frequency_clustering_injection_parameters_filename
        )
        self.time_frequency_clustering_pe_samples_filename = args.time_frequency_clustering_pe_samples_filename
        self.time_frequency_clustering_threshold = args.time_frequency_clustering_threshold
        self.time_frequency_clustering_threshold_type = args.time_frequency_clustering_threshold_type
        self.time_frequency_clustering_time_padding = args.time_frequency_clustering_time_padding
        self.time_frequency_clustering_frequency_padding = args.time_frequency_clustering_frequency_padding
        self.time_frequency_clustering_skypoints = args.time_frequency_clustering_skypoints

        if create_data:
            self.create_data(args)

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

    def _get_interferometers_from_injection_in_gaussian_noise(self):
        # Copy the interferometers
        interferometers = bilby.gw.detector.InterferometerList([ifo.name for ifo in self.interferometers])
        for i in range(len(interferometers)):
            power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
                frequency_array=self.interferometers[i].frequency_array.copy(),
                psd_array=self.interferometers[i].power_spectral_density_array.copy(),
            )
            interferometers[i].power_spectral_density = power_spectral_density
        injection_parameters = self.injection_df.iloc[self.idx].to_dict()
        # Set the strain data from zero noise.
        interferometers.set_strain_data_from_zero_noise(
            sampling_frequency=self.sampling_frequency, duration=self.duration, start_time=self.start_time
        )
        waveform_arguments = self.get_injection_waveform_arguments()
        waveform_generator = self.waveform_generator_class(
            duration=self.duration,
            start_time=self.start_time,
            sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=self.bilby_frequency_domain_source_model,
            parameter_conversion=self.parameter_conversion,
            waveform_arguments=waveform_arguments,
        )
        interferometers.inject_signal(
            waveform_generator=waveform_generator,
            parameters=injection_parameters,
            raise_error=self.enforce_signal_duration,
        )
        return interferometers

    def run_time_frequency_clustering(self):
        logger.info(f"Running time-frequency clustering with method: {self.time_frequency_clustering_method}")
        # Build the strain data for time-frequency clustering
        if self.time_frequency_clustering_method in [
            "data",
            "injection",
            "injection_parameters_file",
            "maxL",
            "maP",
            "random",
        ]:
            if self.time_frequency_clustering_method == "data":
                frequency_domain_strain_array = np.array([ifo.frequency_domain_strain for ifo in self.interferometers])
            elif self.time_frequency_clustering_method in [
                "injection",
                "injection_parameters_file",
                "maxL",
                "maP",
                "random",
            ]:
                if self.time_frequency_clustering_method == "injection":
                    parameters = self.meta_data["injection_parameters"]
                elif self.time_frequency_clustering_method == "injection_parameters_file":
                    if self.time_frequency_clustering_injection_parameters_filename is None:
                        raise NullpolError(
                            "time-frequency-clustering-injection-parameters-filename must be provided when time-frequency-clustering-method = 'injection_parameters_file'."
                        )
                    injection_df = Input.read_injection_file(
                        self.time_frequency_clustering_injection_parameters_filename
                    )
                    if len(injection_df) > 1:
                        logger.warning(
                            f"More than one injections in time-frequency-clustering-injection-parameters-filename={self.time_frequency_clustering_injection_parameters_filename}."
                        )
                        logger.warning("Use the first injection.")
                    parameters = injection_df.iloc[0].to_dict()
                elif self.time_frequency_clustering_method in ["maxL", "maP", "random"]:
                    if self.time_frequency_clustering_pe_samples_filename is None:
                        raise NullpolError(
                            f"time-frequency-clustering-method = {self.time_frequency_clustering_method}, but time-frequency-clustering-pe-samples-filename is not provided."
                        )
                    try:
                        posterior = bilby.core.result.read_in_result(
                            self.time_frequency_clustering_pe_samples_filename
                        ).posterior
                    except Exception:
                        logger.warning("Trying to read the posterior as a .h5 file.")
                        with h5py.File(self.time_frequency_clustering_pe_samples_filename, "r") as f:
                            posterior = pd.DataFrame(
                                f[list(f.keys())[0]]["posterior_samples"][()]
                            )  # Use the first waveform result
                    if self.time_frequency_clustering_method == "maxL":
                        parameters = posterior.loc[posterior["log_likelihood"].idxmax()].to_dict()
                    elif self.time_frequency_clustering_method == "maP":
                        parameters = posterior.loc[
                            (posterior["log_likelihood"] + posterior["log_prior"]).idxmax()
                        ].to_dict()
                    elif self.time_frequency_clustering_method == "random":
                        parameters = posterior.sample().to_dict()
                    else:
                        raise NullpolError(
                            f"Unexpected error with time-frequency-clustering-method = {self.time_frequency_clustering_method}. Contact the developers."
                        )
                    # Remove log_likelihood and log_prior from parameters
                    parameters.pop("log_likelihood", None)
                    parameters.pop("log_prior", None)
                    # Remove tidal parameters except lambda_1 and lambda_2
                    remove_keys = [
                        "lambda_tilde",
                        "delta_lambda_tilde",
                        "lambda_symmetric",
                        "eos_polytrope_gamma_0",
                        "eos_spectral_pca_gamma_0",
                        "eos_v1",
                    ]
                    if (
                        any(key in parameters for key in remove_keys)
                        or "lambda_1" in parameters
                        or "lambda_2" in parameters
                    ):
                        logger.info(
                            f'Found tidal parameters: {",".join([key for key in ["lambda_1", "lambda_2"] if key in parameters]+[key for key in remove_keys if key in parameters])}'
                        )
                        if "lambda_1" not in parameters:
                            raise NullpolError("lambda_1 and lambda_2 must be both provided. lambda_1 is missing.")
                        if "lambda_2" not in parameters:
                            raise NullpolError("lambda_1 and lambda_2 must be both provided. lambda_2 is missing.")
                        for key in remove_keys:
                            if key in parameters:
                                parameters.pop(key, None)
                                logger.warning(f"Removing {key} except lambda_1 and lambda_2.")
                # Generate the mock strain data from the parameters.
                ## Generate a new interferometer list with the same detectors
                logger.info("Generating zero-noise injection data")
                ifos = bilby.gw.detector.InterferometerList([ifo.name for ifo in self.interferometers])
                # Copy the power spectral density
                for i in range(len(ifos)):
                    ifos[i].power_spectral_density = self.interferometers[i].power_spectral_density
                ifos.set_strain_data_from_zero_noise(
                    sampling_frequency=self.sampling_frequency, duration=self.duration, start_time=self.start_time
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
                raise ValueError(f"Unknown time-frequency clustering method {self.time_frequency_clustering_method}")
            time_frequency_filter, sky_maximized_spectrogram = _run_time_frequency_clustering(
                interferometers=self.interferometers,
                frequency_domain_strain_array=frequency_domain_strain_array,
                wavelet_frequency_resolution=self.wavelet_frequency_resolution,
                wavelet_nx=self.wavelet_nx,
                threshold=self.time_frequency_clustering_threshold,
                time_padding=self.time_frequency_clustering_time_padding,
                frequency_padding=self.time_frequency_clustering_frequency_padding,
                skypoints=self.time_frequency_clustering_skypoints,
                return_sky_maximized_spectrogram=True,
                threshold_type=self.time_frequency_clustering_threshold_type,
            )
            # Generate the cumulative distribution of the spectrogram.
            reversed_cumulative_distribution_fig_fname = (
                f"{self.data_directory}/{self.label}_reversed_cumulative_distribution_of_whitened_energy_pixels.png"
            )
            plot_reverse_cumulative_distribution(
                spectrogram=sky_maximized_spectrogram,
                bins=25,
                title="Reversed cumulative distribution of whitened energy pixels",
                savefig=reversed_cumulative_distribution_fig_fname,
            )
            if np.sum(time_frequency_filter) == 0:
                # Generate the plot for diagnostics.
                spectrogram_fig_fname = f"{self.data_directory}/{self.label}_sky_maximized_spectrogram.png"
                plot_spectrogram(
                    spectrogram=sky_maximized_spectrogram,
                    duration=self.interferometers[0].duration,
                    sampling_frequency=self.interferometers[0].sampling_frequency,
                    wavelet_frequency_resolution=self.wavelet_frequency_resolution,
                    frequency_range=(self.minimum_frequency, self.maximum_frequency),
                    t0=self.start_time,
                    title="Sky-maximized Spectrogram",
                    savefig=spectrogram_fig_fname,
                    dpi=100,
                )
                logger.info(f"Saved plot of sky-maximized spectrogram to {spectrogram_fig_fname}.")
                raise NullpolError("The time_frequency_filter is empty. Terminating...")
        elif is_file(self.time_frequency_clustering_method):
            time_frequency_filter = np.load(self.time_frequency_clustering_method)
            sky_maximized_spectrogram = None
            logger.info(f"Loaded time-frequency filter from {self.time_frequency_clustering_method}.")
        else:
            raise NullpolError(
                f"Unrecognized time-frequency-clustering-method = {self.time_frequency_clustering_method}."
            )
        # Generate the sky-maximized spectrogram from data
        sky_maximized_spectrogram_data = compute_sky_maximized_spectrogram(
            interferometers=self.interferometers,
            frequency_domain_strain_array=np.array([ifo.frequency_domain_strain for ifo in self.interferometers]),
            wavelet_frequency_resolution=self.wavelet_frequency_resolution,
            wavelet_nx=self.wavelet_nx,
            skypoints=self.time_frequency_clustering_skypoints,
        )
        self.meta_data["time_frequency_filter"] = time_frequency_filter
        self.meta_data["sky_maximized_spectrogram"] = sky_maximized_spectrogram
        self.meta_data["sky_maximized_spectrogram_data"] = sky_maximized_spectrogram_data
        # Plot
        time_frequency_filter_fig_fname = f"{self.data_directory}/{self.label}_time_frequency_filter_spectrogram.png"
        plot_spectrogram(
            spectrogram=time_frequency_filter,
            duration=self.interferometers[0].duration,
            sampling_frequency=self.interferometers[0].sampling_frequency,
            wavelet_frequency_resolution=self.wavelet_frequency_resolution,
            frequency_range=(self.minimum_frequency, self.maximum_frequency),
            t0=self.start_time,
            title="Time-frequency Filter Spectrogram",
            savefig=time_frequency_filter_fig_fname,
            dpi=100,
        )
        logger.info(f"Saved plot of time-frequency filter spectrogram to {time_frequency_filter_fig_fname}.")
        # Plot
        if sky_maximized_spectrogram is not None:
            spectrogram_fig_fname = f"{self.data_directory}/{self.label}_sky_maximized_spectrogram.png"
            plot_spectrogram(
                spectrogram=sky_maximized_spectrogram,
                duration=self.interferometers[0].duration,
                sampling_frequency=self.interferometers[0].sampling_frequency,
                wavelet_frequency_resolution=self.wavelet_frequency_resolution,
                frequency_range=(self.minimum_frequency, self.maximum_frequency),
                t0=self.start_time,
                title="Sky-maximized Spectrogram",
                savefig=spectrogram_fig_fname,
                dpi=100,
            )
            logger.info(f"Saved plot of sky-maximized spectrogram to {spectrogram_fig_fname}.")
        # Plot
        spectrogram_data_fig_fname = f"{self.data_directory}/{self.label}_sky_maximized_spectrogram_data.png"
        plot_spectrogram(
            spectrogram=sky_maximized_spectrogram_data,
            duration=self.interferometers[0].duration,
            sampling_frequency=self.interferometers[0].sampling_frequency,
            wavelet_frequency_resolution=self.wavelet_frequency_resolution,
            frequency_range=(self.minimum_frequency, self.maximum_frequency),
            t0=self.start_time,
            title="Sky-maximized Spectrogram Data",
            savefig=spectrogram_data_fig_fname,
            dpi=100,
        )
        logger.info(f"Saved plot of sky-maximized spectrogram of data to {spectrogram_data_fig_fname}.")

    def save_data_dump(self):
        """Method to dump the saved data to disk for later analysis"""
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
