from bilby.gw.source import _base_lal_cbc_fd_waveform


def lal_binary_black_hole_non_gr_simple_map(
    frequency_array, mass_1, mass_2, luminosity_distance,
    a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl,
    theta_jn, phase,
    amp_plus=1., amp_cross=1., amp_x=0., amp_y=0., amp_b=0., amp_l=0.,
    **kwargs,
):
    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomPv2', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)
    wf_dict = _base_lal_cbc_fd_waveform(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
        phi_jl=phi_jl, **waveform_kwargs)
    wf_dict['x'] = wf_dict['plus'] * amp_x
    wf_dict['y'] = wf_dict['cross'] * amp_y
    wf_dict['breathing'] = wf_dict['plus'] * amp_b
    wf_dict['longitudinal'] = wf_dict['plus'] * amp_l
    wf_dict['plus'] = wf_dict['plus'] * amp_plus
    wf_dict['cross'] = wf_dict['cross'] * amp_cross
    return wf_dict
