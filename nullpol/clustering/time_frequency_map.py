from ..time_frequency_transform import (transform_wavelet_freq,
                                        transform_wavelet_freq_quadrature)
from ..utility.whiten import whiten_frequency_domain_strain


def construct_time_frequency_map(interferometers, frequency_resolution):
    ndet = len(interferometers)
    Nf = int(interferometers[0].sampling_frequency / 2 / frequency_resolution)
    Nt = int(len(interferometers[0].time_array) / Nf)
    nx = 4
    # Compute the whitened time-frequency array
    whitened_frequency_domain_strain_0 = whiten_frequency_domain_strain(interferometers[0])
    whitened_time_frequency_domain_strain_0 = transform_wavelet_freq(whitened_frequency_domain_strain_0,
                                                                     Nf,
                                                                     Nt,
                                                                     nx)
    whitened_time_frequency_domain_strain_quadrature_0 = transform_wavelet_freq_quadrature(whitened_frequency_domain_strain_0,
                                                                                           Nf,
                                                                                           Nt,
                                                                                           nx)
    combined_power = whitened_time_frequency_domain_strain_0 ** 2 + whitened_time_frequency_domain_strain_quadrature_0 ** 2
    for i in range(1, ndet):
        whitened_frequency_domain_strain_i = whiten_frequency_domain_strain(interferometers[i])
        whitened_time_frequency_domain_strain_i = transform_wavelet_freq(whitened_frequency_domain_strain_i,
                                                                         Nf,
                                                                         Nt,
                                                                         nx)
        whitened_time_frequency_domain_strain_quadrature_i = transform_wavelet_freq_quadrature(whitened_frequency_domain_strain_i,
                                                                                               Nf,
                                                                                               Nt,
                                                                                               nx)
        combined_power += (whitened_time_frequency_domain_strain_i ** 2 + whitened_time_frequency_domain_strain_quadrature_i ** 2)
    return combined_power