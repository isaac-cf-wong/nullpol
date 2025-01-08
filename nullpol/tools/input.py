from importlib import import_module
import bilby
from bilby_pipe.input import Input as BilbyInput
import bilby_pipe.utils
from bilby_pipe.utils import (convert_string_to_dict,
                              BilbyPipeError,
                              strip_quotes)
import numpy as np
from ..utility import logger
from ..likelihood import Chi2TimeFrequencyLikelihood
from .. import prior as nullpol_prior
import inspect


bilby_pipe.utils.logger  = logger

class Input(BilbyInput):
    def __init__(self, args, unknown_args, print_msg=True):
        super(Input, self).__init__(args=args,
                                    unknown_args=unknown_args,
                                    print_msg=print_msg)
        self.polarization_modes = args.polarization_modes
        self.polarization_basis = args.polarization_basis
        self.wavelet_frequency_resolution = args.wavelet_frequency_resolution
        self.wavelet_nx = args.wavelet_nx
        self.simulate_psd_nsample = args.simulate_psd_nsample
        self.calibration_correction_type = args.calibration_correction_type

        # Waveform, source model and likelihood
        self.reference_frequency = args.reference_frequency
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
        self.duration = args.duration
        self.trigger_time = args.trigger_time
        self.post_trigger_duration = args.post_trigger_duration

    @property
    def priors(self):
        """Read in and compose the prior at run-time"""
        if getattr(self, "_priors", None) is None:
            self._priors = self._get_priors()
            self._add_default_extrinsic_priors()
        return self._priors

    @priors.setter
    def priors(self, priors):
        self._priors = priors

    def _add_default_extrinsic_priors(self):
        if 'ra' not in self._priors:
            self._priors['ra'] = bilby.core.prior.Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary="periodic")
            logger.info(f"Added missing prior for ra: {self._priors['ra']}")
        if 'dec' not in self._priors:
            self._priors['dec'] = bilby.core.prior.Cosine(name="dec")
            logger.info(f"Added missing prior for dec: {self._priors['dec']}")
        if 'psi' not in self._priors:
            self._priors['psi'] = bilby.core.prior.Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic')
            logger.info(f"Added missing prior for psi: {self._priors['psi']}")

    @property
    def likelihood(self):
        self.search_priors = self.priors.copy()
        likelihood_kwargs = dict(
            interferometers=self.interferometers,
            wavelet_frequency_resolution=self.wavelet_frequency_resolution,
            wavelet_nx=self.wavelet_nx,
            polarization_modes=self.polarization_modes,
            polarization_basis=self.polarization_basis,
            time_frequency_filter=f"{self.label}_time_frequency_filter.npy",
            simulate_psd_nsample=self.simulate_psd_nsample,
            calibration_marginalization=self.calibration_marginalization,
            calibration_lookup_table=self.calibration_lookup_table,
            calibration_psd_lookup_table=self.calibration_psd_lookup_table,
            number_of_response_curves=self.number_of_response_curves,
            priors=self.search_priors,            
        )
        if self.likelihood_type == "Chi2TimeFrequencyLikelihood":
            Likelihood = Chi2TimeFrequencyLikelihood
        elif "." in self.likelihood_type:
            split_path = self.likelihood_type.split(".")
            module = ".".join(split_path[:-1])
            likelihood_class = split_path[-1]
            Likelihood = getattr(import_module(module), likelihood_class)
            likelihood_kwargs.update(self.extra_likelihood_kwargs)            
        else:
            raise ValueError(f"Unknown Likelihood class {self.likelihood_type}")

        likelihood_kwargs = {
            key: likelihood_kwargs[key]
            for key in likelihood_kwargs
            if key in inspect.getfullargspec(Likelihood.__init__).args
        }

        logger.debug(
            f"Initialise likelihood {Likelihood} with kwargs: \n{likelihood_kwargs}"
        )
        likelihood = Likelihood(**likelihood_kwargs)        

        # If requested, use a zero likelihood: for testing purposes
        if self.likelihood_type == "zero":
            logger.debug("Using a ZeroLikelihood")
            likelihood = bilby.core.likelihood.ZeroLikelihood(likelihood)

        return likelihood
    
    @property
    def combined_default_prior_dicts(self):
        d = nullpol_prior.__dict__.copy()
        return d

    @property
    def calibration_psd_lookup_table(self):
        return getattr(self, "_calibration_psd_lookup_table", None)

    @calibration_psd_lookup_table.setter
    def calibration_psd_lookup_table(self, lookup):
        if isinstance(lookup, str):
            lookup = convert_string_to_dict(lookup)
        self._calibration_psd_lookup_table = lookup

    @property
    def calibration_correction_type(self):
        return getattr(self, "_calibration_correction_type", None)

    @calibration_correction_type.setter
    def calibration_correction_type(self, correction_type):
        self._calibration_correction_type = correction_type

    @property
    def polarization_modes(self):
        return getattr(self, '_polarization_modes', None)
    
    @polarization_modes.setter
    def polarization_modes(self, modes):
        if modes is None:
            raise BilbyPipeError("No polarization-modes input")
        if isinstance(modes, list):
            modes = ",".join(modes)
        if isinstance(modes, str) is False:
            raise BilbyPipeError(f"polarization-modes input {modes} not understood")
        # Remove square brackets
        modes = modes.replace("[", "").replace("]", "")
        # Remove added quotes
        modes = strip_quotes(modes)
        # Replace multiple spaces with a single space
        modes = " ".join(modes.split())
        # Spaces can be either space or comma in input, convert to comma
        modes = modes.replace(" ,", ",").replace(", ", ",").replace(" ", ",")

        modes = modes.split(",")

        self._polarization_modes = modes

    @property
    def polarization_basis(self):
        return getattr(self, '_polarization_basis', None)
    
    @polarization_basis.setter
    def polarization_basis(self, basis):
        self._polarization_basis = basis
        if basis is not None:
            logger.debug(f"Polarization basis set to {basis}")
        else:
            self._polarization_basis = self.polarization_modes
            logger.debug(f"Polarization basis set to default value of {self._polarization_basis}")

    @property
    def simulate_psd_nsample(self):
        return getattr(self, "_simulate_psd_nsample", None)

    @simulate_psd_nsample.setter
    def simulate_psd_nsample(self, nsample):
        self._simulate_psd_nsample = nsample     

    @property
    def wavelet_nx(self):
        return getattr(self, "_wavelet_nx", None)
    
    @wavelet_nx.setter
    def wavelet_nx(self, wavelet_nx):
        self._wavelet_nx = wavelet_nx
        if wavelet_nx is not None:
            logger.debug(f"wavelet_nx set to {wavelet_nx}")
        else:
            self._wavelet_nx = 4.
            logger.debug(f"wavelet_nx set to default value of 4.")               

    @property
    def wavelet_frequency_resolution(self):
        return getattr(self, "_wavelet_frequency_resolution", None)

    @wavelet_frequency_resolution.setter
    def wavelet_frequency_resolution(self, resolution):
        self._wavelet_frequency_resolution = resolution

    def get_injection_waveform_arguments(self):
        """Get the dict of the waveform arguments needed for creating injections.

        Defaults the injection-waveform-approximant to waveform-approximant, if
        no injection-waveform-approximant provided. Note that the default
        waveform-approximant is `IMRPhenomPv2`.
        """
        if self.injection_waveform_approximant is None:
            self.injection_waveform_approximant = self.waveform_approximant
        waveform_arguments = self.get_default_waveform_arguments()
        waveform_arguments["waveform_approximant"] = self.injection_waveform_approximant
        waveform_arguments["numerical_relativity_file"] = self.numerical_relativity_file
        return waveform_arguments

    @property
    def injection_parameters(self):
        return self._injection_parameters

    @injection_parameters.setter
    def injection_parameters(self, injection_parameters):
        self._injection_parameters = injection_parameters
        if self.calibration_prior is not None:
            for key in self.calibration_prior:
                if key not in injection_parameters:
                    if "frequency" in key:
                        injection_parameters[key] = self.calibration_prior[key].peak
                    else:
                        injection_parameters[key] = 0
        self.meta_data["injection_parameters"] = injection_parameters