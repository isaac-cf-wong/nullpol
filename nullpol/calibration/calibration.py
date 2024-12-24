from bilby.gw.detector.calibration import (read_calibration_file,
                                           write_calibration_file,
                                           _generate_calibration_draws,
                                           Precomputed,
                                           Recalibrate)
import os
import numpy as np
import pandas as pd
from pycbc.types.frequencyseries import FrequencySeries
from ..utility import logger
from ..psd import (simulate_psd_from_psd,
                   get_pycbc_psd)

def build_calibration_lookup(
    interferometers,
    wavelet_frequency_resolution,
    simulate_psd_nsample,
    wavelet_nx=4.,
    lookup_files=None,
    psd_lookup_files=None,
    priors=None,
    number_of_response_curves=1000,
    starting_index=0,
):
    if lookup_files is None and priors is None:
        raise ValueError(
            "One of calibration_lookup_table or priors must be specified for "
            "building calibration marginalization lookup table."
        )
    elif lookup_files is None:
        lookup_files = dict()
        psd_lookup_files = dict()

    draws = dict()
    parameters = dict()
    psd_draws = dict()
    for interferometer in interferometers:
        name = interferometer.name
        frequencies = interferometer.frequency_array
        frequencies = frequencies[interferometer.frequency_mask]
        filename = lookup_files.get(name, f"{name}_calibration_file.h5")
        psd_filename = psd_lookup_files.get(name, f"{name}_calibration_psd_file.h5")

        if os.path.exists(filename):
            draws[name], parameters[name] = read_calibration_file(
                filename,
                frequencies,
                number_of_response_curves,
                starting_index,
            )
            if os.path.exists(psd_filename):
                psd_draws[name] = read_calibration_psd_file(psd_filename)
            else:
                psd_draws[name] = _generate_calibration_psd_draws(interferometer=interferometer,
                                                                  calibration_draws=draws[name],
                                                                  wavelet_frequency_resolution=wavelet_frequency_resolution,
                                                                  simulate_psd_nsample=simulate_psd_nsample,
                                                                  wavelet_nx=wavelet_nx)
                write_calibration_psd_file(psd_filename, psd_draws[name])
        elif isinstance(interferometer.calibration_model, Precomputed):
            model = interferometer.calibration_model
            idxs = np.arange(number_of_response_curves, dtype=int) + starting_index
            draws[name] = model.curves[idxs]
            parameters[name] = pd.DataFrame(model.parameters.iloc[idxs])
            parameters[name][model.prefix] = idxs
            if os.path.exists(psd_filename):
                psd_draws[name] = read_calibration_psd_file(psd_filename)
            else:
                psd_draws[name] = _generate_calibration_psd_draws(interferometer=interferometer,
                                                                  calibration_draws=draws[name],
                                                                  wavelet_frequency_resolution=wavelet_frequency_resolution,
                                                                  simulate_psd_nsample=simulate_psd_nsample,
                                                                  wavelet_nx=wavelet_nx)
                write_calibration_psd_file(psd_filename, psd_draws[name])
        else:
            if priors is None:
                raise ValueError(
                    "Priors must be passed to generate calibration response curves "
                    "for cubic spline."
                )
            draws[name], parameters[name] = _generate_calibration_draws(
                interferometer=interferometer,
                priors=priors,
                n_curves=number_of_response_curves,
            )
            write_calibration_file(filename, frequencies, draws[name], parameters[name])
            psd_draws[name] = _generate_calibration_psd_draws(interferometer=interferometer,
                                                              calibration_draws=draws[name],
                                                              wavelet_frequency_resolution=wavelet_frequency_resolution,
                                                              simulate_psd_nsample=simulate_psd_nsample,
                                                              wavelet_nx=wavelet_nx)
            write_calibration_psd_file(psd_filename, psd_draws[name])

        interferometer.calibration_model = Recalibrate()

    return draws, parameters, psd_draws

def read_calibration_psd_file(psd_filename):
    import tables
    logger.info("Reading calibration psd draws from {psd_filename}")
    calibration_psd_file = tables.open_file(psd_filename, 'r')
    psd = calibration_psd_file.root.psd.psd_draws[:]
    calibration_psd_file.close()

    if len(psd.dtype) != 0:
        psd = psd.view(np.float64)

    return psd

def write_calibration_psd_file(psd_filename, psd_draws):
    import tables
    logger.info(f"Writing calibration psd draws to {psd_filename}")
    calibration_psd_file = tables.open_file(psd_filename, 'w')
    psd_group = calibration_psd_file.create_group(calibration_psd_file.root, 'psd')

    # Save output
    calibration_psd_file.create_carray(psd_group, 'psd_draws', obj=psd_draws)
    calibration_psd_file.close()

def _generate_calibration_psd_draws(interferometer,
                                    calibration_draws,
                                    wavelet_frequency_resolution,
                                    simulate_psd_nsample,
                                    wavelet_nx):
    delta_f = interferometer.frequency_array[1]-interferometer.frequency_array[0]
    psd_draws = []
    for calibration_draw in calibration_draws:
        psd_array = interferometer.power_spectral_density_array.copy()
        psd_array[interferometer.frequency_mask] = psd_array[interferometer.frequency_mask] / np.abs(calibration_draw)**2
        psd_pycbc = get_pycbc_psd(psd_array, delta_f=delta_f)
        psd_draws.append(simulate_psd_from_psd(psd_pycbc,interferometer.duration,interferometer.sampling_frequency,wavelet_frequency_resolution,simulate_psd_nsample,wavelet_nx))
    return np.array(psd_draws)