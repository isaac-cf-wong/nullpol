def frequency_domain_unconstrained_waveform(freuency_array,
                                            right_ascension,
                                            declination,
                                            polarization_angle,
                                            relative_amplitude=None,
                                            relative_phase=None,
                                            **kwargs):
    
    interferometers = kwargs['interferometers']
    minimum_frequency = kwargs['minimum_frequency']
    maximum_frequency = kwargs['maximum_frequency']
    pass