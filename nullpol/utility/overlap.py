import numpy as np
from pycbc.detector import Detector


def compute_overlap_between_nontensorial_and_tensorial_polarizations(detectors,
                                                                     ra_array,
                                                                     dec_array,
                                                                     gpstime):
    vector_overlap = []
    scalar_overlap = []
    pycbc_detectors = [Detector(det) for det in detectors]
    for ra, dec in zip(ra_array, dec_array):
        fp, fc = np.array([det.antenna_pattern(ra, dec, 0., gpstime, polarization_type='tensor') for det in pycbc_detectors])
        fx, fy = np.array([det.antenna_pattern(ra, dec, 0., gpstime, polarization_type='vector') for det in pycbc_detectors])
        fb, fl = np.array([det.antenna_pattern(ra, dec, 0., gpstime, polarization_type='scalar') for det in pycbc_detectors])