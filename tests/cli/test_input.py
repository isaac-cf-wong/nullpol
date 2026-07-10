"""Tests for likelihood construction from CLI input."""

from __future__ import annotations

from bilby_pipe.main import parse_args

from nullpol.cli import input as input_module
from nullpol.cli.data_analysis import DataAnalysisInput
from nullpol.cli.parser import create_nullpol_parser


class _KeywordOnlyLikelihood:
    """Capture the arguments supplied by :attr:`Input.likelihood`."""

    def __init__(
        self,
        interferometers,
        wavelet_frequency_resolution,
        wavelet_nx,
        polarization_modes,
        *args,
        polarization_basis=None,
        time_frequency_filter=None,
        **kwargs,
    ):
        self.interferometers = interferometers
        self.wavelet_frequency_resolution = wavelet_frequency_resolution
        self.wavelet_nx = wavelet_nx
        self.polarization_modes = polarization_modes
        self.polarization_basis = polarization_basis
        self.time_frequency_filter = time_frequency_filter
        self.extra_kwargs = kwargs


def test_analysis_config_forwards_keyword_only_parameters(monkeypatch, tmp_path):
    """A configured analysis forwards its normalized basis to the likelihood."""
    config_file = tmp_path / "analysis.ini"
    config_file.write_text(
        "polarization-modes = pc\npolarization-basis = p\n",
        encoding="utf-8",
    )
    args, unknown_args = parse_args([str(config_file)], create_nullpol_parser(top_level=False))

    assert unknown_args == []
    assert args.polarization_modes == ["pc"]
    assert args.polarization_basis == ["p"]

    interferometers = object()
    time_frequency_filter = object()
    priors = object()
    inputs = object.__new__(DataAnalysisInput)
    inputs._interferometers = interferometers
    inputs.search_priors = priors
    inputs.wavelet_frequency_resolution = 16.0
    inputs.wavelet_nx = 4.0
    inputs.polarization_modes = args.polarization_modes
    inputs.polarization_basis = args.polarization_basis
    inputs.meta_data = {"time_frequency_filter": time_frequency_filter}
    inputs.likelihood_type = args.likelihood_type
    inputs.extra_likelihood_kwargs = args.extra_likelihood_kwargs

    assert inputs.polarization_modes == "pc"
    assert inputs.polarization_basis == "p"

    monkeypatch.setattr(input_module, "Chi2TimeFrequencyLikelihood", _KeywordOnlyLikelihood)

    likelihood = input_module.Input.likelihood.fget(inputs)

    assert likelihood.interferometers is interferometers
    assert likelihood.wavelet_frequency_resolution == 16.0
    assert likelihood.wavelet_nx == 4.0
    assert likelihood.polarization_modes == "pc"
    assert likelihood.polarization_basis == "p"
    assert likelihood.time_frequency_filter is time_frequency_filter
    assert likelihood.extra_kwargs == {}
