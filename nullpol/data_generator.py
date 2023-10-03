import numpy as np
#import bilby
#from bilby.gw.detector import InterferometerList
import bilby
from bilby.gw.detector import PowerSpectralDensity
from bilby.gw.waveform_generator import WaveformGenerator
from bilby.gw.source import lal_binary_black_hole
from bilby_pipe.utils import is_a_power_of_2

from .logging import logger
from .detector.networks import *


class DataGenerator:
    def __init__(self,
                 detectors,
                 sampling_frequency,
                 duration,
                 start_time=0,
                 waveform_generator_class=WaveformGenerator,
                 frequency_domain_source_model="lal_binary_black_hole",
                 psd_dict=None,
                 zero_noise=False,
                 waveform_arguments=None,
                 generation_seed=None):
        self.detectors = detectors
        self.sampling_frequency = sampling_frequency
        self.duration = duration
        self.start_time = start_time
        self.waveform_generator_class = waveform_generator_class
        self.frequency_domain_source_model = frequency_domain_source_model
        self.psd_dict = psd_dict
        self.zero_noise = zero_noise
        self.calibration_prior = None
        self.waveform_arguments = waveform_arguments
        self.generation_seed = generation_seed
        self.meta_data = {}

    @property
    def frequency_domain_source_model(self):
        """ String of which frequency domain source model to use """

        return self._frequency_domain_source_model

    @frequency_domain_source_model.setter
    def frequency_domain_source_model(self, frequency_domain_source_model):
        self._frequency_domain_source_model = frequency_domain_source_model        
        
    @property
    def bilby_frequency_domain_source_model(self):
        """
        The bilby function to pass to the waveform_generator

        This can be a function defined in an external package.
        """

        if self.frequency_domain_source_model in bilby.gw.source.__dict__.keys():
            model = self._frequency_domain_source_model
            logger.info(f"Using the {model} source model")

            return bilby.gw.source.__dict__[model]
        elif "." in self.frequency_domain_source_model:
            return get_function_from_string_path(self._frequency_domain_source_model)
        else:
            raise BilbyPipeError(
                f"No source model {self._frequency_domain_source_model} found."
            )    
    
    @property
    def sampling_frequency(self):
        return self._sampling_frequency

    @sampling_frequency.setter
    def sampling_frequency(self, sampling_frequency):
        if is_a_power_of_2(sampling_frequency) is False:
            logger.warning(
                "Sampling frequency {} not a power of 2, this can cause problems".format(
                    sampling_frequency
                )
            )
        self._sampling_frequency = sampling_frequency    

    @property
    def parameter_conversion(self):
        if "binary_neutron_star" in self.frequency_domain_source_model:
            return bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters
        elif "binary_black_hole" in self.frequency_domain_source_model:
            return bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters
        else:
            return None        
        
    @property
    def generation_seed(self):
        return self._generation_seed
    
    @generation_seed.setter
    def generation_seed(self, generation_seed):
        """Sets the generation seed.

        If no generation seed has been provided, a random seed between 1 and 1e6 is
        selected.

        If a seed is provided, it is used as the base seed and all generation jobs will
        have their seeds set as {generation_seed = base_seed + job_idx}.

        NOTE: self.idx must not be None

        Parameters
        ----------
        generation_seed: int or None

        """

        if generation_seed is None:
            generation_seed = np.random.randint(1, 1e6)
        self._generation_seed = generation_seed
        np.random.seed(generation_seed)
        logger.info(f"Generation seed set to {generation_seed}")

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
        
    def _set_psd_from_file(self, ifo):
        psd_file = self.psd_dict[ifo.name]
        logger.info(f"Setting {ifo.name} PSD from file {psd_file}")
        ifo.power_spectral_density = (
            PowerSpectralDensity.from_power_spectral_density_file(psd_file=psd_file)
        )        
        
    def _set_interferometers_from_gaussian_noise(self):
        """ Method to generate the interferometers data from Gaussian noise """

        ifos = bilby.gw.detector.InterferometerList(self.detectors)
        
        if self.psd_dict is not None:
            for ifo in ifos:
                if ifo.name in self.psd_dict.keys():
                    self._set_psd_from_file(ifo)

        if self.zero_noise:
            logger.info("Setting strain data from zero noise")
            ifos.set_strain_data_from_zero_noise(
                sampling_frequency=self.sampling_frequency,
                duration=self.duration,
                start_time=self.start_time,
            )
        else:
            logger.info("Simulating strain data from psd-colored noise")
            ifos.set_strain_data_from_power_spectral_densities(
                sampling_frequency=self.sampling_frequency,
                duration=self.duration,
                start_time=self.start_time,
            )

        self.interferometers = ifos
        
    def _set_interferometers_from_injection_in_gaussian_noise(self):
        """ Method to generate the interferometers data from an injection in Gaussian noise """

        #self.injection_parameters = self.injection_df.iloc[self.idx].to_dict()
        #logger.info("Injecting waveform with ")

        for prop in [
            "minimum_frequency",
            "maximum_frequency",
            "trigger_time",
            "start_time",
            "duration",
        ]:
            #logger.info(f"{prop} = {getattr(self, prop)}")
            pass

        self._set_interferometers_from_gaussian_noise()

        #logger.info(f"Using waveform arguments: {waveform_arguments}")
        waveform_generator = self.waveform_generator_class(
            duration=self.duration,
            start_time=self.start_time,
            sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=self.bilby_frequency_domain_source_model,
            parameter_conversion=self.parameter_conversion,
            waveform_arguments=self.waveform_arguments,
        )

        self.interferometers.inject_signal(
            waveform_generator=waveform_generator, parameters=self.injection_parameters
        )

    def generate_injection(self, parameters):
        self.injection_parameters = parameters
        self._set_interometers_from_injection_in_gaussian_noise()

        return self.interferometers
