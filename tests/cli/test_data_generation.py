"""Tests for building :class:`DataGenerationInput` from a configuration."""

from __future__ import annotations

import numpy as np
from bilby_pipe.main import parse_args

from nullpol.cli.data_generation import DataGenerationInput, create_generation_parser

TRIGGER_TIME = 1126259642.413

INJECTION_PARAMETERS = {
    "mass_1": 36.0,
    "mass_2": 29.0,
    "a_1": 0.4,
    "a_2": 0.3,
    "tilt_1": 0.5,
    "tilt_2": 1.0,
    "phi_12": 1.7,
    "phi_jl": 0.3,
    "luminosity_distance": 400.0,
    "theta_jn": 0.4,
    "psi": 2.659,
    "phase": 1.3,
    "geocent_time": TRIGGER_TIME,
    "ra": 1.375,
    "dec": -1.2108,
    "amp_p": 1.0,
    "amp_c": 1.0,
    "amp_b": 0.5,
    "amp_l": 0.0,
    "amp_x": 0.0,
    "amp_y": 0.0,
}


def _parse_config(tmp_path, extra=""):
    injection_file = tmp_path / "injections.dat"
    injection_file.write_text(
        " ".join(INJECTION_PARAMETERS) + "\n" + " ".join(repr(value) for value in INJECTION_PARAMETERS.values()) + "\n",
        encoding="utf-8",
    )
    config_file = tmp_path / "config.ini"
    config_file.write_text(
        f"label = test\n"
        f"outdir = {tmp_path / 'outdir'}\n"
        "detectors = [H1, L1, V1]\n"
        "duration = 4\n"
        "sampling-frequency = 1024\n"
        f"trigger-time = {TRIGGER_TIME}\n"
        "injection = True\n"
        f"injection-file = {injection_file}\n"
        "gaussian-noise = True\n"
        "n-simulation = 1\n"
        "generation-seed = 1234\n"
        "injection-waveform-approximant = IMRPhenomPv2\n"
        "injection-frequency-domain-source-model = "
        "nullpol.simulation.source.lal_binary_black_hole_non_gr_simple_map\n"
        "reference-frequency = 50\n"
        "time-frequency-clustering-method = injection\n"
        "polarization-modes = pc\n"
        "polarization-basis = p\n" + extra,
        encoding="utf-8",
    )
    args, unknown_args = parse_args([str(config_file), "--idx", "0"], create_generation_parser())
    assert unknown_args == []
    return args, unknown_args


def test_injection_in_gaussian_noise_generates_data(tmp_path):
    """An injection into Gaussian noise runs the upstream injection path to completion."""
    args, unknown_args = _parse_config(tmp_path)

    data = DataGenerationInput(args, unknown_args)

    assert [ifo.name for ifo in data.interferometers] == ["H1", "L1", "V1"]
    assert data.injection_parameters["mass_1"] == INJECTION_PARAMETERS["mass_1"]
    assert data.meta_data["injection_parameters"]["amp_b"] == INJECTION_PARAMETERS["amp_b"]
    for ifo in data.interferometers:
        assert ifo.meta_data["optimal_SNR"] > 0
        assert np.all(np.isfinite(ifo.frequency_domain_strain))


def test_upstream_data_generation_attributes_follow_config(tmp_path):
    """Attributes read by bilby_pipe's data generation are taken from the config."""
    args, unknown_args = _parse_config(
        tmp_path,
        "waveform-generator-constructor-dict = {'foo': 1}\n"
        "fetch-open-data-kwargs = {'cache': True}\n"
        "psd-auto-gating-configuration = {'threshold': 50}\n"
        "segment-auto-gating-configuration = {'threshold': 60}\n"
        "reweighting-configuration = {'bar': 2}\n"
        "cosmology = Planck15\n",
    )

    data = DataGenerationInput(args, unknown_args, create_data=False)

    assert data.waveform_generator_class_ctor_args == "{'foo': 1}"
    assert data.fetch_open_data_kwargs == {"cache": True}
    assert data.psd_auto_gating_configuration == {"threshold": 50}
    assert data.segment_auto_gating_configuration == {"threshold": 60}
    assert data.reweighting_configuration == {"bar": 2}
    assert data.cosmology.name == "Planck15"
