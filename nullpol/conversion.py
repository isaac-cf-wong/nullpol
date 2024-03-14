def empty_conversion_func (sample, likelihood=None, priors=None, npool=None):
    """A function to bypass the conversion of parameters for bilby_pipe compatibility.

    Parameters
    ==========
    sample: dict or pandas.DataFrame
        Samples to fill in with extra parameters, this may be either an
        injection or posterior samples.
    likelihood: bilby.gw.likelihood.GravitationalWaveTransient, optional
        GravitationalWaveTransient used for sampling, used for waveform and
        likelihood.interferometers.
    priors: dict, optional
        Dictionary of prior objects, used to fill in non-sampled parameters.

    Returns
    -------
    dict or pandas.DataFrame: The input sample, unchanged.
    """
    return sample
