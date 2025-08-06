from __future__ import annotations

import configparser
from ast import literal_eval

import bilby_pipe


def read_bilby_ini_file(file: str) -> dict:
    """
    Read a bilby ini file and return the contents as a dictionary

    Parameters
    ==========
    file : str
        The path to the bilby ini file

    Returns
    =======
    dict
        The contents of the ini file as a dictionary
    """
    with open(file) as f:
        file_content = f.read()

    # add dummy section header
    file_content = '[dummy]\n' + file_content

    # read the file with the parser
    config = configparser.ConfigParser()
    config.read_string(file_content)

    return dict(config['dummy'])

def convert_string_to_dict(string):
    try:
        return bilby_pipe.utils.convert_string_to_dict(string)
    except (bilby_pipe.utils.BilbyPipeError, AttributeError):
        return string

def fill_in_pol_specific_metadata(analysis, corresponding_analysis):
    """
    For pol analysis, fill in fields in metadata other than result list.

    Parameters
    ==========
    analysis: Asimov production for given event
        corresponding_analysis - equivalent subanalysis as stored in cbcflow

    Returns
    =======
    analysis_output: dict
        Dictionary used to update cbcflow with new information.
    """
    analysis_output = {}
    analysis_output["AnalysisSoftware"] = str(analysis.pipeline)
    analysis_output["Description"] = "Polarization analyses"
    return analysis_output

def bilby_config_to_asimov(config_name):
    """
    Read a bilby ini file and return content in asimov ledger compatible form.

    Parameters
    ==========
    config_name : str
        The path to the bilby ini file

    Returns
    =======
    dict
        The contents of the ini file as a dictionary
    """
    config = read_bilby_ini_file(config_name)
    output_dict = {}

    ifos = bilby_pipe.utils.convert_detectors_input(config['detectors'])
    output_dict['ifos'] = ifos
    output_dict['event time'] = config['trigger-time']
    if isinstance(output_dict['event time'], str):
        output_dict['event time'] = literal_eval(output_dict['event time'])
    output_dict['psds'] = convert_string_to_dict(config['psd-dict'])

    # fill in quality
    quality = {}
    waveform_freq = None
    for freq in ['minimum-frequency', 'maximum-frequency']:
        freq_value = convert_string_to_dict(config[freq])
        if isinstance(freq_value, dict):
            if freq == 'minimum-frequency':
                if 'waveform' in freq_value:
                    waveform_freq = freq_value.pop('waveform')
            quality[freq.replace('-', ' ')] = freq_value
        else:
            quality[freq.replace('-', ' ')] = {ifo: freq_value for ifo in ifos}
    output_dict['quality'] = quality

    # fill in data
    data = {}

    fields = {'spline-calibration-envelope-dict': 'calibration',
              'data-dict': 'data files',
              'channel-dict': 'channels',
              'frame-type-dict': 'frame types',
              'duration': 'segment length',
              'data-format': 'format'}
    for name_ini, name_asimov in fields.items():
        if name_ini in config:
            value = convert_string_to_dict(config[name_ini])
            if value is not None:
                data[name_asimov] = value
    if isinstance(data['segment length'], str):
        data['segment length'] = literal_eval(data['segment length'])

    output_dict['data'] = data

    # copy likelihood fields
    likelihood = {}
    if waveform_freq is not None:
        likelihood['start frequency'] = waveform_freq

    fields = {'time-reference': 'time reference',
              'reference-frame': 'reference frame'}
    for name_ini, name_asimov in fields.items():
        if name_ini in config:
            value = config[name_ini]
            if value is not None:
                likelihood[name_asimov] = value

    fields = {'post-trigger-duration': 'post trigger time',
              'sampling-frequency': 'sample rate',
              'psd-length': 'psd length',
              'tukey-roll-off': 'roll off time'}
    for name_ini, name_asimov in fields.items():
        if name_ini in config:
            value = convert_string_to_dict(config[name_ini])
            if value is not None:
                if isinstance(value, str):
                    value = literal_eval(value)
                likelihood[name_asimov] = value

    marginalization = {}
    fields = {'distance-marginalization': 'distance',
              'phase-marginalization': 'phase',
              'time-marginalization': 'time',
              'calibration-marginalization': 'calibration'}
    for name_ini, name_asimov in fields.items():
        if name_ini in config:
            value = convert_string_to_dict(config[name_ini])
            if value is not None:
                if isinstance(value, str):
                    value = literal_eval(value)
                marginalization[name_asimov] = value
    likelihood['marginalization'] = marginalization
    output_dict['likelihood'] = likelihood

    # copy waveform settings
    waveform = {}
    fields = {'reference-frequency': 'reference frequency',
              'waveform-approximant': 'approximant',
              'pn-spin-order': 'pn spin order',
              'pn-tidal-order': 'pn tidal order',
              'pn-phase-order': 'pn phase order',
              'pn-amplitude-order': 'pn amplitude order',
              'waveform-arguments-dict': 'arguments',
              'mode-array': 'mode array'}
    for name_ini, name_asimov in fields.items():
        if name_ini in config:
            if name_ini == 'waveform-approximant':
                value = config[name_ini]
            else:
                value = convert_string_to_dict(config[name_ini])
            if value is not None:
                if name_ini != 'mode-array' or value[0] is not None:
                    waveform[name_asimov] = value
    for name_asimov in ['pn amplitude order',
                        'pn phase order',
                        'pn spin order',
                        'pn tidal order',
                        'reference frequency']:
        if isinstance(waveform[name_asimov], str):
            waveform[name_asimov] = literal_eval(waveform[name_asimov])
    output_dict['waveform'] = waveform

    # read prior
    prior = {}

    if config['prior-dict'] != 'None':
        prior_dict = bilby_pipe.utils.convert_prior_string_input(
            config['prior-dict'])
        prior_dict = {
            key.replace('-', '_'): value for key, value in prior_dict.items()}
    elif config['prior-file'] is not None:
        try:
            with open(config['prior-file']) as f:
                temp_dict = f.read()
            lines = temp_dict.split('\n')
            prior_dict = {}
            for line in lines:
                if not any(c.isalpha() for c in line):
                    continue
                else:
                    line.replace('{', '')
                    line.replace('}', '')
                    line.replace(" ", "")
                    elements = line.split("=")
                    key = elements[0].replace(" ", "")
                    val = "=".join(elements[1:]).strip()
                prior_dict[key] = val
        except OSError:  # file might have been deleted
            prior_dict = None
    else:
        prior_dict = None
    if prior_dict is not None:
        fields = {'geocent_time': 'geocentric time',
                  'chirp_mass': 'chirp mass',
                  'mass_ratio': 'mass ratio',
                  'total_mass': 'total mass',
                  'mass_1': 'mass 1',
                  'mass_2': 'mass 2',
                  'a_1': 'a 1',
                  'a_2': 'a 2',
                  'tilt_1': 'tilt 1',
                  'tilt_2': 'tilt 2',
                  'phi_12': 'phi 12',
                  'phi_jl': 'phi jl',
                  'lambda_1': 'lambda 1',
                  'lambda_2': 'lambda 2',
                  'luminosity_distance': 'luminosity distance'}
        for name_ini, name_asimov in fields.items():
            if name_ini in prior_dict:
                subprior = prior_dict[name_ini].replace(' ', '')
                param_prior = {}
                prior_type, prior_rest = subprior.split('(', 1)
                param_prior['type'] = prior_type

                prior_rest = prior_rest.rsplit(')', 1)[0]
                prior_rest = prior_rest.split(',')
                for parameter in prior_rest:
                    key, value = parameter.split('=', 1)
                    if key not in ['name', 'unit', 'latex_label']:
                        if key == 'boundary':
                            param_prior[key] = convert_string_to_dict(value)
                        else:
                            try:
                                param_prior[key] = float(value)
                            except ValueError:
                                param_prior[key] = value

                prior[name_asimov] = param_prior

    fields = {'default-prior': 'default',
              'calibration-prior-boundary': 'boundary'}
    for name_ini, name_asimov in fields.items():
        if name_ini in config:
            value = config[name_ini]
            if value is not None:
                if name_ini == 'calibration-prior-boundary':
                    prior['calibration'] = {name_asimov: value}
                else:
                    prior[name_asimov] = value
    output_dict['priors'] = prior
    return output_dict

def deep_update(mapping, *updating_mappings):
    # updates dict recursively
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping and
                isinstance(updated_mapping[k], dict) and
                isinstance(v, dict)
            ):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping
