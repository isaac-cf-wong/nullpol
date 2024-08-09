import numpy as np
from nullpol.clustering import clustering


def clustering(interferometers, padding_time, padding_freq, threshold):
    energy_map = np.abs(interferometers[0].whitened_time_frequency_domain_strain_array)**2 + np.abs(interferometers[0].whitened_time_frequency_domain_quadrature_strain_array)**2
    for i in range(1, len(interferometers)):
        energy_map += np.abs(interferometers[i].whitened_time_frequency_domain_strain_array)**2 + np.abs(interferometers[i].whitened_time_frequency_domain_quadrature_strain_array)**2
    dt = interferometers[0].duration / interferometers[0].Nt
    df = interferometers[0].sampling_frequency / 2 / interferometers[0].Nf
    return clustering(energy_map, dt, df, padding_time, padding_freq, threshold)    