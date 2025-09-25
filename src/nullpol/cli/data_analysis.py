# pylint: disable=duplicate-code  # Legitimate shared calibration configuration patterns
from __future__ import annotations

import signal
import sys
from argparse import Namespace

import numpy as np

# Configure matplotlib backend before importing libraries that use it
# fmt: off
import matplotlib  # isort:skip
matplotlib.use("agg")
# fmt: on

# Imports below must be after matplotlib backend configuration
# pylint: disable=wrong-import-position
import bilby
import bilby_pipe.utils
from bilby_pipe.data_analysis import DataAnalysisInput as BInput
from bilby_pipe.main import parse_args
from bilby_pipe.utils import CHECKPOINT_EXIT_CODE

from .. import log_version_information
from ..analysis.antenna_patterns import encode_polarization
from ..analysis.antenna_patterns import relative_amplification_factor_map
from ..analysis.result import PolarizationResult
from ..utils import NullpolError, logger
from .input import Input
from .parser import create_nullpol_parser

# pylint: enable=wrong-import-position

bilby_pipe.utils.logger = logger


def sighandler(signum, frame):  # pylint: disable=unused-argument
    logger.info("Performing periodic eviction")
    sys.exit(CHECKPOINT_EXIT_CODE)


# pylint: disable=too-many-instance-attributes
class DataAnalysisInput(BInput, Input):
    """Handles user-input for the data analysis script."""

    def __init__(self, args: Namespace, unknown_args: list, test: bool = False):
        # pylint: disable=super-init-not-called  # Using Input.__init__ directly for clarity
        """Initializer.

        Args:
            args (tuple): A tuple of arguments.
            unknown_args (tuple): A tuple of unknown arguments.
            test (bool, optional): _description_. Defaults to False.
        """
        Input.__init__(self, args, unknown_args)

        # Generic initialization
        self.meta_data = dict()
        self.result = None

        self.injection_parameters = None

        # Admin arguments
        self.ini = args.ini
        self.scheduler = args.scheduler
        self.periodic_restart_time = args.periodic_restart_time
        self.request_cpus = args.request_cpus
        self.run_local = args.local

        # Naming arguments
        self.outdir = args.outdir
        self.label = args.label
        self.result_format = args.result_format

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
        self.duration = args.duration

        # Likelihood
        self.likelihood_type = args.likelihood_type
        self.polarization_modes = args.polarization_modes
        self.polarization_basis = args.polarization_basis
        self.extra_likelihood_kwargs = args.extra_likelihood_kwargs

        # Calibration
        self.calibration_model = args.calibration_model
        self.calibration_correction_type = args.calibration_correction_type
        self.spline_calibration_envelope_dict = args.spline_calibration_envelope_dict
        self.spline_calibration_amplitude_uncertainty_dict = args.spline_calibration_amplitude_uncertainty_dict
        self.spline_calibration_phase_uncertainty_dict = args.spline_calibration_phase_uncertainty_dict
        self.spline_calibration_nodes = args.spline_calibration_nodes
        self.calibration_prior_boundary = args.calibration_prior_boundary

        # Injection arguments
        self.injection_waveform_approximant = args.injection_waveform_approximant

        if test is False:
            self._load_data_dump()

    @property
    def polarization_modes(self):
        return getattr(self, "_polarization_modes", None)

    @polarization_modes.setter
    def polarization_modes(self, modes):
        if isinstance(modes, str):
            self._polarization_modes = modes
        elif isinstance(modes, list):
            if len(modes) == 1:
                self._polarization_modes = modes[0]
            else:
                raise NullpolError("Unsupported polarization-modes input: " f"{modes}")
        else:
            raise NullpolError("Unsupported polarization-modes input: " f"{modes}")

    @property
    def polarization_basis(self):
        return getattr(self, "_polarization_basis", None)

    @polarization_basis.setter
    def polarization_basis(self, modes):
        if isinstance(modes, str):
            self._polarization_basis = modes
        elif isinstance(modes, list):
            if len(modes) == 1:
                self._polarization_basis = modes[0]
            else:
                raise NullpolError("Unsupported polarization-basis input: " f"{modes}")
        else:
            raise NullpolError("Unsupported polarization-basis input: " f"{modes}")

    def _validate_polarization_model_setting(self):
        supported_modes = ["b", "c", "b", "l", "x", "y"]
        # Check whether the modes are supported.
        for mode in self.polarization_modes:
            if mode not in supported_modes:
                raise NullpolError(
                    f"Unsupported mode `{mode}` in " "polarization-modes = " f"{self.polarization_modes}"
                )
        for mode in self.polarization_basis:
            if mode not in supported_modes:
                raise NullpolError(f"Unsupported mode `{mode}` in polarization-basis = {self.polarization_basis}")
        # Check whether the basis is part of the model.
        for mode in self.polarization_basis:
            if mode not in self.polarization_modes:
                raise NullpolError(f"Basis mode `{mode}` is not in polarization-modes = {self.polarization_basis}")

    @property
    def result_class(self):
        """The nullpol result class to store results in"""
        return PolarizationResult

    def get_likelihood_and_priors(self) -> tuple:
        """Read in the likelihood and prior from the data dump

        This reads in the data dump values and reconstructs the likelihood and
        priors. Note, care must be taken to use the "search_priors" which differ
        from the true prior when using marginalization

        Returns:
            tuple: The bilby likelihood and priors
        """

        priors = self.data_dump.priors_class(self.data_dump.priors_dict)
        self.priors = priors.copy()
        self._add_default_relative_polarization_prior()
        likelihood = self.likelihood
        priors = self.search_priors
        return likelihood, priors

    def _add_default_relative_polarization_prior(self):
        # Encode the polarization modes
        _polarization_modes, polarization_basis, polarization_derived = encode_polarization(
            self.polarization_modes, self.polarization_basis
        )

        # Obtain the keywords
        relative_polarization_parameters = relative_amplification_factor_map(
            polarization_basis=polarization_basis, polarization_derived=polarization_derived
        ).flatten()
        prior_remove_list = []
        for prior in self.priors:
            if prior[:10] == "amplitude_" and prior[10:] not in relative_polarization_parameters:
                prior_remove_list.append(prior)
            elif prior[:6] == "phase_" and prior[6:] not in relative_polarization_parameters:
                prior_remove_list.append(prior)

        # Remove the redundant parameters
        for prior_key in prior_remove_list:
            logger.info(f"Removed irrelevant prior: {self.priors[prior_key]}")
            self.priors.pop(prior_key)

        # Add missing prior
        missing_priors = {}
        for label in relative_polarization_parameters:
            amplitude_name = f"amplitude_{label}"
            if amplitude_name not in self.priors:
                missing_priors[amplitude_name] = bilby.core.prior.Uniform(
                    name=amplitude_name, minimum=0.0, maximum=1.0, latex_label=f"$A_{{{label}}}$"
                )
                logger.info(f"Added missing relative polarization prior: {missing_priors[amplitude_name]}")
            phase_name = f"phase_{label}"
            if phase_name not in self.priors:
                missing_priors[phase_name] = bilby.core.prior.Uniform(
                    name=phase_name,
                    minimum=0.0,
                    maximum=2.0 * np.pi,
                    latex_label=rf"$\phi_{{{label}}}$",
                    boundary="periodic",
                )
                logger.info(f"Added missing relative polarization prior: {missing_priors[phase_name]}")
        self.priors.update(missing_priors)

    def run_sampler(self):
        if self.scheduler.lower() == "condor" and not self.run_local:
            signal.signal(signal.SIGALRM, handler=sighandler)
            signal.alarm(self.periodic_restart_time)

        likelihood, priors = self.get_likelihood_and_priors()

        logger.info(f"Analyzing with polarization modes: {self.polarization_modes}")
        logger.info(f"Analyzing with polarization basis: {self.polarization_basis}")

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


def create_analysis_parser():
    """Data analysis parser creation"""
    return create_nullpol_parser(top_level=False)


def main():
    """Data analysis main logic"""
    args, unknown_args = parse_args(
        sys.argv[1:],
        create_analysis_parser(),
    )
    log_version_information()
    analysis = DataAnalysisInput(args, unknown_args)
    analysis.run_sampler()
    if analysis.reweighting_configuration is not None:
        analysis.reweight_result()
    logger.info("Run completed")
