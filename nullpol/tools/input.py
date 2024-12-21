import bilby
from bilby_pipe.input import Input as BilbyInput
import bilby_pipe.utils
from bilby_pipe.utils import convert_string_to_dict
from importlib import import_module
import inspect
from ..utility import logger
from ..likelihood import Chi2TimeFrequencyLikelihood


bilby_pipe.utils.logger  = logger

class Input(BilbyInput):
    def __init__(self, args, unknown_args, print_msg=True):
        super(Input, self).__init__(args=args,
                                    unknown_args=unknown_args,
                                    print_msg=print_msg)
        self.polarization_modes = args.polarization_modes
        self.polarization_basis = args.polarization_modes
        self.wavelet_frequency_resolution = args.wavelet_frequency_resolution
        self.wavelet_nx = args.wavelet_nx
        self.calibration_correction_type = args.calibration_correction_type

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
        print(likelihood_kwargs)
        likelihood = Likelihood(**likelihood_kwargs)        

        # If requested, use a zero likelihood: for testing purposes
        if self.likelihood_type == "zero":
            logger.debug("Using a ZeroLikelihood")
            likelihood = bilby.core.likelihood.ZeroLikelihood(likelihood)

        return likelihood

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
        self._polarization_modes = modes
        if modes is not None:
            logger.debug(f"Polarization modes set to {modes}")
        else:
            self._polarization_modes = 'pc'
            logger.debug(f"Polarization modes set to default value of {self._polarization_modes}")

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