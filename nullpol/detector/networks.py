import numpy as np
import bilby.gw.detector


@property
def time_domain_strain_array(self):
    if self._time_domain_strain_array is None:
        nifo = len(self)
        nfreq = len(self[0].time_domain_strain)
        self._time_domain_strain_array = np.zeros((nifo, nfreq), dtype=self[0].time_domain_strain[0].dtype)

        for i in range(nifo):
            self._time_domain_strain_array[i,:] = self[i].time_domain_strain            

    return self._time_domain_strain_array

@property
def frequency_domain_strain_array(self):
    if self._frequency_domain_strain_array is None:
        nifo = len(self)
        nfreq = len(self[0].frequency_domain_strain)
        self._frequency_domain_strain_array = np.zeros((nifo, nfreq), dtype=self[0].frequency_domain_strain[0].dtype)

        for i in range(nifo):
            self._frequency_domain_strain_array[i,:] = self[i].frequency_domain_strain            

    return self._frequency_domain_strain_array

@property
def time_frequency_domain_strain_array(self):
    if self._frequency_domain_strain_array is None:
        nifo = len(self)
        ntime, nfreq = self[0].time_frequency_domain_strain.shape
        self._frequency_domain_strain_array = np.zeros((nifo, ntime, nfreq), dtype=self[0].time_frequency_domain_strain[0, 0].dtype)

        for i in range(nifo):
            self._time_frequency_domain_strain_array[i,:] = self[i].time_frequency_domain_strain

    return self._time_frequency_domain_strain_array

@property
def whitened_frequency_domain_strain_array(self):
    if self._whitened_frequency_domain_strain_array is None:
        nifo = len(self)
        nfreq = len(self[0].frequency_domain_strain)
        self._whitened_frequency_domain_strain_array = np.zeros((nifo, nfreq), dtype=self[0].frequency_domain_strain[0].dtype)

        for i in range(nifo):
            df = self[i].frequency_array[1] - self[i].frequency_array[0]
            whitening_factor = 1/np.sqrt(self[i].power_spectral_density_array/(2*df))
            self._whitened_frequency_domain_strain_array[i,:] = self[i].frequency_domain_strain*whitening_factor

    return self._whitened_frequency_domain_strain_array


bilby.gw.detector.InterferometerList._time_domain_strain_array = None
bilby.gw.detector.InterferometerList.time_domain_strain_array = time_domain_strain_array
bilby.gw.detector.InterferometerList._frequency_domain_strain_array = None
bilby.gw.detector.InterferometerList.frequency_domain_strain_array = frequency_domain_strain_array
bilby.gw.detector.InterferometerList._time_frequency_domain_strain_array = None
bilby.gw.detector.InterferometerList.time_frequency_domain_strain_array = time_frequency_domain_strain_array
bilby.gw.detector.InterferometerList._whitened_frequency_domain_strain_array = None
bilby.gw.detector.InterferometerList.whitened_frequency_domain_strain_array = whitened_frequency_domain_strain_array
