import os
from bilby.core.prior.dict import PriorDict
from ..utility import logger


DEFAULT_PRIOR_DIR = os.path.join(os.path.dirname(__file__), 'prior_files')

class PolarizationPriorDict(PriorDict):
    def __init__(self, dictionary=None, filename=None):
        if dictionary is None and filename is None:
            fname = 'polarization.prior'
            filename = os.path.join(DEFAULT_PRIOR_DIR, fname)
            logger.info('No prior given, using default polarization priors in {}.'.format(filename))
        elif filename is not None:
            if not os.path.isfile(filename):
                filename = os.path.join(DEFAULT_PRIOR_DIR, filename)
        super(PolarizationPriorDict, self).__init__(dictionary=dictionary, filename=filename)

    def validate_prior(self, **kwargs):
        return True
