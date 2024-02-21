# The following code is copied from
# https://git.ligo.org/ania.liu/millilensing/-/blob/main/src/prior.py
# and modified to fit the needs of this project.

import numpy as np 
from bilby.core.prior import Prior 

class DiscreteUniform(Prior):
    def __init__(self, name=None, latex_label=None, unit=None):
        """
        DiscreteUniform 
        ------------

        discrete uniform sampling for Morse phase, 
        generates samples using inverse CDF method. 

        The relative Morse factor can take values of 0, 0.5 and 1 corresponding to 
        types of the images. The later image always has a equal or higher Morse factor. 
        """

        super(DiscreteUniform, self).__init__(
        name=name,
        latex_label=latex_label,
        minimum = 0,
        maximum = 2
        )


    def rescale(self, val):
            """
            continuous interval from 0 to 1 mapped 
            to discrete distribution in 0, 0.5, 1 
            """
            return np.floor((self.maximum+1)*val)/2

    def prob(self, val):
            """
            take 1/3 probability for each value 
            """
            return ((val >= 0) & (val <= self.maximum/2.))/float(self.maximum + 1) * (np.modf(2*val)[0] == 0).astype(int)

    def cdf(self, val):
            """
            cumulative density funstion
            """
            return (val <= self.maximum/2.)*(np.floor(2*val)+1)/float(self.maximum+1) +(val > self.maximum/2.)
