import numpy as np
import bilby
from nullpol.wdm.wavelet_transform import (transform_wavelet_freq,
                                        transform_wavelet_freq_quadrature)

@property
def time_frequency_bandwidth(self):
    return self._time_frequency_bandwidth

@time_frequency_bandwidth.setter
def time_frequency_bandwidth(self, value):
    self._time_frequency_bandwidth = value
    self._Nf = int(self.sampling_frequency / 2 / value)
    self._Nt = int(len(self) / self._Nf)

@property
def Nf(self):
    if self._Nf is None:
        raise ValueError('Nf is undefined.')
    return self._Nf

@property
def Nt(self):
    if self._Nt is None:
        raise ValueError('Nt is undefined.')
    return self._Nt

@property
def nx(self):
    if self._nx is None:
        raise ValueError('nx is undefined.')
    return self._nx

@nx.setter
def nx(self, value):
    self._nx = value

@property
def whitened_time_frequency_domain_strain_array(self):
    if self._whitened_time_frequency_domain_strain_array is None:
        # Perform time-frequency transform
        self._whitened_time_frequency_domain_strain_array = transform_wavelet_freq(self.whitened_frequency_domain_strain_array,
                                                                                   self.Nf,
                                                                                   self.Nt,
                                                                                   self.nx)
    return self._whitened_time_frequency_domain_strain_array

@property
def whitened_time_frequency_domain_quadrature_strain_array(self):
    if self._whitened_time_frequency_domain_quadrature_strain_array is None:
        # Perform time-frequency transform
        self._whitened_time_frequency_domain_quadrature_strain_array = transform_wavelet_freq_quadrature(self.whitened_frequency_domain_strain_array,
                                                                                                         self.Nf,
                                                                                                         self.Nt,
                                                                                                         self.nx)
    return self._whitened_time_frequency_domain_quadrature_strain_array

bilby.gw.detector.strain_data.InterferometerStrainData._time_frequency_bandwidth = None
bilby.gw.detector.strain_data.InterferometerStrainData.time_frequency_bandwidth = time_frequency_bandwidth
bilby.gw.detector.strain_data.InterferometerStrainData._Nf = None
bilby.gw.detector.strain_data.InterferometerStrainData._Nt = None
bilby.gw.detector.strain_data.InterferometerStrainData._nx = None
bilby.gw.detector.strain_data.InterferometerStrainData.Nf = Nf
bilby.gw.detector.strain_data.InterferometerStrainData.Nt = Nt
bilby.gw.detector.strain_data.InterferometerStrainData.nx = nx
bilby.gw.detector.strain_data.InterferometerStrainData._whitened_time_frequency_domain_strain_array = None
bilby.gw.detector.strain_data.InterferometerStrainData.whitened_time_frequency_domain_strain_array = whitened_time_frequency_domain_strain_array
bilby.gw.detector.strain_data.InterferometerStrainData._whitened_time_frequency_domain_quadrature_strain_array = None
bilby.gw.detector.strain_data.InterferometerStrainData.whitened_time_frequency_domain_quadrature_strain_array = whitened_time_frequency_domain_quadrature_strain_array