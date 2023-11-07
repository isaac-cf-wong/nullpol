# The following code is copied from
# https://git.ligo.org/ania.liu/millilensing/-/blob/main/src/prior.py

import numpy as np 
from bilby.core.prior import Prior 

class DiscreteUniform(Prior):
    def __init__(self, name=None, latex_label=None, unit=None):
        """
        DiscreteUniform 
        ------------

        discrete uniform sampling for Morse phase, 
        generates samples using inverse CDF method. 

        The Morse factor can take values of 0, 0.5 and 1 corresponding to 
        Type I, II and III images, respectively.
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
             
class DiscreteInteger(Prior):

    def __init__(self, name=None, latex_label=None, unit=None, minimum=1, N=-1):
        """
        DiscreteInteger
        ----------
        Sample uniformly over a discrete set of integers 
        from minimum,...,N+minimum-1 using inverse CDF method.
        """

        if isinstance(N, int) and N>= 0:
            self.N = N

        else:
            raise ValueError("Not a positive integer")

        super(DiscreteInteger, self).__init__(
                name = name,
                latex_label = latex_label,
                minimum = int(minimum),
                maximum = int((self.N + minimum) -1),
                boundary = None
                ) 

    def rescale(self, val):
        if isinstance(val, float):
            return int(np.floor(self.N*val) + self.minimum)
        else:
            return (np.floor(self.N*val) + self.minimum).astype(int)

    def prob(self, val):
        return ((val >= self.minimum) & (val <= self.maximum))/float(self.N) * (np.modf(val)[0] == 0).astype(int)

    def cdf(delf, val):
        return (val <= self.maximum) * (np.floor(val) - self.minimum + 1)/float(self.N) + (val > self.maximum)
