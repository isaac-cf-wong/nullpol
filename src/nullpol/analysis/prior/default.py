from __future__ import annotations

import os

from bilby.core.prior.dict import PriorDict

from ...utils import logger

# Directory containing default prior configuration files
DEFAULT_PRIOR_DIR = os.path.join(os.path.dirname(__file__), "prior_files")


class PolarizationPriorDict(PriorDict):
    """Prior dictionary specialized for null stream polarization analysis.

    Extends Bilby's PriorDict to provide convenient access to default prior
    distributions for polarization parameters. Automatically loads default
    polarization priors when no specific configuration is provided.

    The default polarization prior file includes:
        - dec: Cosine distribution for declination (isotropic sky distribution)
        - ra: Uniform distribution for right ascension (0 to 2π, periodic)
        - psi: Uniform distribution for polarization angle (0 to π, periodic)

    This class handles the loading of prior files from the default directory
    and provides validation for polarization-specific parameters.

    Attributes:
        Inherits all attributes from bilby.core.prior.dict.PriorDict.

    Example:
        >>> # Use default polarization priors
        >>> priors = PolarizationPriorDict()

        >>> # Load custom prior file
        >>> priors = PolarizationPriorDict(filename='my_priors.prior')

        >>> # Use custom prior dictionary
        >>> custom_priors = {'amplitude_pp': Uniform(0, 1)}
        >>> priors = PolarizationPriorDict(dictionary=custom_priors)
    """

    def __init__(self, dictionary=None, filename=None):
        """Initialize the polarization prior dictionary.

        Args:
            dictionary (dict, optional): Dictionary of prior distributions.
                If provided, these priors will be used directly. Keys should
                be parameter names and values should be Bilby Prior objects.
            filename (str, optional): Path to a prior configuration file.
                If not an absolute path, the file will be searched for in
                the default prior directory. If None and dictionary is None,
                the default 'polarization.prior' file will be loaded.

        Note:
            If both dictionary and filename are None, the default polarization
            prior file ('polarization.prior') will be automatically loaded
            from the package's prior_files directory.
        """
        if dictionary is None and filename is None:
            fname = "polarization.prior"
            filename = os.path.join(DEFAULT_PRIOR_DIR, fname)
            logger.info(f"No prior given, using default polarization priors in {filename}.")
        elif filename is not None:
            if not os.path.isfile(filename):
                filename = os.path.join(DEFAULT_PRIOR_DIR, filename)
        super().__init__(dictionary=dictionary, filename=filename)

    def validate_prior(self, **kwargs):  # pylint: disable=unused-argument
        """Validate the prior distributions for polarization analysis.

        Performs validation checks on the prior distributions to ensure they
        are appropriate for gravitational wave polarization analysis. This
        method can be extended to include specific validation rules for
        polarization parameters.

        Args:
            **kwargs: Additional keyword arguments for validation. Currently
                unused but provided for future extensibility.

        Returns:
            bool: Always returns True in the current implementation. Future
                versions may implement specific validation logic and return
                False if validation fails.

        Note:
            This method currently provides a placeholder for future validation
            logic. It can be extended to check parameter bounds, correlations,
            or other constraints specific to polarization analysis.
        """
        return True
