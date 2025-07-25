from __future__ import annotations

from bilby.gw.source import _base_lal_cbc_fd_waveform


def lal_binary_black_hole_non_gr_simple_map(
    frequency_array, mass_1, mass_2, luminosity_distance,
    a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl,
    theta_jn, phase,
    amp_p=1., amp_c=1., amp_x=0., amp_y=0., amp_b=0., amp_l=0.,
    **kwargs,
):
    """Generate binary black hole waveform with simple amplitude scaling for non-GR polarizations.

    Computes frequency-domain gravitational waveforms for binary black hole coalescence
    including non-General Relativity polarization modes through simple amplitude scaling.
    The additional polarization modes (vector x/y, breathing, longitudinal) are constructed
    by scaling the standard plus and cross polarizations.

    Args:
        frequency_array (numpy.ndarray): Array of frequencies in Hz.
        mass_1 (float): Primary mass in solar masses.
        mass_2 (float): Secondary mass in solar masses.
        luminosity_distance (float): Luminosity distance in Mpc.
        a_1 (float): Dimensionless spin magnitude of primary.
        tilt_1 (float): Tilt angle of primary spin in radians.
        phi_12 (float): Azimuthal angle between spins in radians.
        a_2 (float): Dimensionless spin magnitude of secondary.
        tilt_2 (float): Tilt angle of secondary spin in radians.
        phi_jl (float): Azimuthal angle between total angular momentum and orbital angular momentum in radians.
        theta_jn (float): Inclination angle in radians.
        phase (float): Phase at reference frequency in radians.
        amp_p (float, optional): Amplitude scaling factor for plus polarization. Defaults to 1.
        amp_c (float, optional): Amplitude scaling factor for cross polarization. Defaults to 1.
        amp_x (float, optional): Amplitude scaling factor for vector x polarization. Defaults to 0.
        amp_y (float, optional): Amplitude scaling factor for vector y polarization. Defaults to 0.
        amp_b (float, optional): Amplitude scaling factor for breathing polarization. Defaults to 0.
        amp_l (float, optional): Amplitude scaling factor for longitudinal polarization. Defaults to 0.
        **kwargs: Additional waveform generation arguments passed to LAL waveform function.

    Returns:
        dict: Dictionary containing frequency-domain polarizations with keys:
            'plus', 'cross', 'x', 'y', 'breathing', 'longitudinal'.
            Each value is a complex numpy array matching frequency_array shape.

    Note:
        This implements a simplified model where non-GR modes are constructed by
        scaling the GR modes: x/y scale h_plus, breathing scales h_plus,
        longitudinal scales h_cross. This is not physically motivated but serves
        as a phenomenological test.
    """
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
    wf_dict['longitudinal'] = wf_dict['cross'] * amp_l
    wf_dict['plus'] = wf_dict['plus'] * amp_p
    wf_dict['cross'] = wf_dict['cross'] * amp_c
    return wf_dict
