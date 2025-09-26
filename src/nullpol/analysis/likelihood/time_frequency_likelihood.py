from __future__ import annotations

from bilby.core.likelihood import Likelihood

from ..data_context import TimeFrequencyDataContext
from ..antenna_patterns import AntennaPatternProcessor
from ..null_stream import NullStreamCalculator


class TimeFrequencyLikelihood(Likelihood):
    """A time-frequency likelihood class with modular architecture.

    This class uses composition to separate concerns:
    - Data management is handled by TimeFrequencyDataContext
    - Antenna pattern operations are handled by AntennaPatternProcessor
    - Null stream computations are handled by NullStreamCalculator

    For low-level data access, use: likelihood.data_context.property_name
    For antenna pattern operations, use: likelihood.antenna_pattern_processor.method_name
    For null stream computations, use: likelihood.null_stream_calculator.method_name
    For high-level likelihood operations, use the methods on this class directly.

    Args:
        interferometers (list): List of interferometers.
        wavelet_frequency_resolution (float): The frequency resolution of the wavelet transform.
        wavelet_nx (int): The number of points in the wavelet transform.
        polarization_modes (list): List of polarization modes.
        polarization_basis (list): List of polarization basis.
        time_frequency_filter (str): The time-frequency filter.
    """

    def __init__(
        self,
        interferometers,
        wavelet_frequency_resolution,
        wavelet_nx,
        polarization_modes,
        *args,  # pylint: disable=unused-argument
        polarization_basis=None,
        time_frequency_filter=None,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(dict())

        # Create the data context component - handles all data management
        self.data_context = TimeFrequencyDataContext(
            interferometers=interferometers,
            wavelet_frequency_resolution=wavelet_frequency_resolution,
            wavelet_nx=wavelet_nx,
            time_frequency_filter=time_frequency_filter,
        )

        # Initialize antenna pattern processor for polarization computations
        self.antenna_pattern_processor = AntennaPatternProcessor(
            polarization_modes=polarization_modes,
            polarization_basis=polarization_basis,
            interferometers=interferometers,
        )

        # Initialize null stream calculator for null stream computations
        self.null_stream_calculator = NullStreamCalculator()

        # Initialize the normalization constant
        self._noise_log_likelihood_value = None

        # Marginalization
        self._marginalized_parameters = []

    @property
    def interferometers(self):
        """A list of interferometers.

        Returns:
            bilby.gw.detector.InterferometerList: A list of interferometers.
        """
        return self.data_context.interferometers

    def log_likelihood(self):
        """Log likelihood.

        Raises:
            NotImplementedError: This should be implemented in a subclass.
        """
        raise NotImplementedError("The log_likelihood method must be implemented in a subclass.")

    def _calculate_noise_log_likelihood(self):
        """Calculate noise log likelihood.
        This should be implemented in a subclass.

        Raises:
            NotImplementedError: This should be implemented in a subclass.
        """
        raise NotImplementedError("The _calculate_noise_log_likelihood method must be implemented in a subclass.")

    def noise_log_likelihood(self):
        """
        Compute the noise log likelihood.

        Returns:
            float: The noise log likelihood.
        """
        if self._noise_log_likelihood_value is None:
            self._noise_log_likelihood_value = self._calculate_noise_log_likelihood()
        return self._noise_log_likelihood_value
