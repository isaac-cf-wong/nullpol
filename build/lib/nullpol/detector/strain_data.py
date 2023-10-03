import numpy as np
from bilby.gw.detector.strain_data import (
    InterferometerStrainData as BilbyInterferometerStrainData
)


class InterferometerStrainData(BilbyInterferometerStrainData):
    """ Strain data for an interferometer """
    def __init__(self, minimum_frequency=0., maximum_frequency=np.inf,
                 roll_off=0.2, notch_list=None):
        """Initiate an InterferometerStrainData object

        The initialized object contains no data, this should be added using one
        of the `set_from..` methods.

        Parameters
        ==========
        minimum_frequency: float
            Minimum frequency to analyze for detector. Default to 0.
        maximum_frequency: float
            Maximum frequency to analyze for detector. Default to np.inf.
        roll_off: float
            The roll-off (in seconds) used in the Tukey window, default=0.2s.
            This corresponds to alpha * duration / 2 for scipy tukey window.
        notch_list: bilby.gw.detector.strain_data.NotchList
            A list of notches
        """
        super(InterferometerStrainData, self).__init__(
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            roll_off=roll_off,
            notch_list=notch_list)
        self._time_frequency_domain_strain = None

    @property
    def time_frequency_domain_strain(self):
        """ The time domain strain data

        Returns
        =======
        time_domain_strain: array_like
            The time domain strain data
        """
        pass
