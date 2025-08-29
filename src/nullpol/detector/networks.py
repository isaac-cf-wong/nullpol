from __future__ import annotations

import math

import bilby.gw.detector
import numpy as np
from bilby.core.utils import logger


@property
def time_domain_strain_array(self):
    """Array of time domain strain data from all interferometers.

    Returns:
        numpy.ndarray: Time domain strain array with shape (n_detectors, n_samples).
    """
    if self._time_domain_strain_array is None:
        nifo = len(self)
        nfreq = len(self[0].time_domain_strain)
        self._time_domain_strain_array = np.zeros((nifo, nfreq), dtype=self[0].time_domain_strain[0].dtype)

        for i in range(nifo):
            self._time_domain_strain_array[i, :] = self[i].time_domain_strain

    return self._time_domain_strain_array


@property
def frequency_domain_strain_array(self):
    """Array of frequency domain strain data from all interferometers.

    Returns:
        numpy.ndarray: Frequency domain strain array with shape (n_detectors, n_frequencies).
    """
    if self._frequency_domain_strain_array is None:
        nifo = len(self)
        nfreq = len(self[0].frequency_domain_strain)
        self._frequency_domain_strain_array = np.zeros((nifo, nfreq), dtype=self[0].frequency_domain_strain[0].dtype)

        for i in range(nifo):
            self._frequency_domain_strain_array[i, :] = self[i].frequency_domain_strain

    return self._frequency_domain_strain_array


@property
def time_frequency_domain_strain_array(self):
    """Array of time-frequency domain strain data from all interferometers.

    Returns:
        numpy.ndarray: Time-frequency domain strain array with shape (n_detectors, n_time, n_frequencies).
    """
    if self._frequency_domain_strain_array is None:
        nifo = len(self)
        ntime, nfreq = self[0].time_frequency_domain_strain.shape
        self._frequency_domain_strain_array = np.zeros(
            (nifo, ntime, nfreq), dtype=self[0].time_frequency_domain_strain[0, 0].dtype
        )

        for i in range(nifo):
            self._time_frequency_domain_strain_array[i, :] = self[i].time_frequency_domain_strain

    return self._time_frequency_domain_strain_array


@property
def whitened_frequency_domain_strain_array(self):
    """Array of whitened frequency domain strain data from all interferometers.

    Returns:
        numpy.ndarray: Whitened frequency domain strain array with shape (n_detectors, n_frequencies).
    """
    if self._whitened_frequency_domain_strain_array is None:
        nifo = len(self)
        nfreq = len(self[0].frequency_domain_strain)
        self._whitened_frequency_domain_strain_array = np.zeros(
            (nifo, nfreq), dtype=self[0].frequency_domain_strain[0].dtype
        )

        for i in range(nifo):
            df = self[i].frequency_array[1] - self[i].frequency_array[0]
            whitening_factor = 1 / np.sqrt(self[i].power_spectral_density_array / (2 * df))
            self._whitened_frequency_domain_strain_array[i, :] = self[i].frequency_domain_strain * whitening_factor

    return self._whitened_frequency_domain_strain_array


def _check_interferometers(self):
    """Verify IFOs 'duration', 'sampling_frequency' are the same.

    If the above attributes are not the same, then the attributes are checked to
    see if they are the same up to 5 decimal places.

    If both checks fail, then a ValueError is raised.
    """
    consistent_attributes = ["duration", "sampling_frequency"]
    for attribute in consistent_attributes:
        x = [getattr(interferometer.strain_data, attribute) for interferometer in self]
        try:
            if not all(y == x[0] for y in x):
                ifo_strs = [
                    "{ifo}[{attribute}]={value}".format(
                        ifo=ifo.name,
                        attribute=attribute,
                        value=getattr(ifo.strain_data, attribute),
                    )
                    for ifo in self
                ]
                raise ValueError(
                    "The {} of all interferometers are not the same: {}".format(attribute, ", ".join(ifo_strs))
                )
        except ValueError as e:
            if not all(math.isclose(y, x[0], abs_tol=1e-5) for y in x):
                raise ValueError(e)
            else:
                logger.warning(e)


bilby.gw.detector.InterferometerList._time_domain_strain_array = None
bilby.gw.detector.InterferometerList.time_domain_strain_array = time_domain_strain_array
bilby.gw.detector.InterferometerList._frequency_domain_strain_array = None
bilby.gw.detector.InterferometerList.frequency_domain_strain_array = frequency_domain_strain_array
bilby.gw.detector.InterferometerList._time_frequency_domain_strain_array = None
bilby.gw.detector.InterferometerList.time_frequency_domain_strain_array = time_frequency_domain_strain_array
bilby.gw.detector.InterferometerList._whitened_frequency_domain_strain_array = None
bilby.gw.detector.InterferometerList.whitened_frequency_domain_strain_array = whitened_frequency_domain_strain_array
bilby.gw.detector.InterferometerList._check_interferometers = _check_interferometers
