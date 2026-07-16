"""Tests for likelihood construction from CLI input."""

from __future__ import annotations

import types

import pytest
from bilby_pipe.main import parse_args

from nullpol.cli import input as input_module
from nullpol.cli.data_analysis import DataAnalysisInput
from nullpol.cli.parser import create_nullpol_parser
from nullpol.utils import NullpolError


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


class _StrictLikelihood:
    """Likelihood without ``**kwargs``; unknown extras must be dropped, not forwarded."""

    def __init__(
        self,
        interferometers,
        wavelet_frequency_resolution,
        wavelet_nx,
        polarization_modes,
        polarization_basis=None,
        time_frequency_filter=None,
        priors=None,
    ):
        self.interferometers = interferometers
        self.wavelet_frequency_resolution = wavelet_frequency_resolution
        self.wavelet_nx = wavelet_nx
        self.polarization_modes = polarization_modes
        self.polarization_basis = polarization_basis
        self.time_frequency_filter = time_frequency_filter
        self.priors = priors


def _build_inputs(args, *, interferometers, time_frequency_filter, priors, likelihood_type=None):
    """Assemble a :class:`DataAnalysisInput` wired for :attr:`Input.likelihood`."""
    inputs = object.__new__(DataAnalysisInput)
    inputs._interferometers = interferometers
    inputs.search_priors = priors
    inputs.wavelet_frequency_resolution = 16.0
    inputs.wavelet_nx = 4.0
    inputs.polarization_modes = args.polarization_modes
    inputs.polarization_basis = args.polarization_basis
    inputs.meta_data = {"time_frequency_filter": time_frequency_filter}
    inputs.likelihood_type = likelihood_type if likelihood_type is not None else args.likelihood_type
    inputs.extra_likelihood_kwargs = args.extra_likelihood_kwargs
    return inputs


def _parse_config(tmp_path, body):
    config_file = tmp_path / "analysis.ini"
    config_file.write_text(body, encoding="utf-8")
    args, unknown_args = parse_args([str(config_file)], create_nullpol_parser(top_level=False))
    assert unknown_args == []
    return args


def test_analysis_config_forwards_keyword_only_parameters(monkeypatch, tmp_path):
    """A configured analysis forwards its normalized basis to the likelihood."""
    args = _parse_config(
        tmp_path,
        "polarization-modes = pc\n"
        "polarization-basis = p\n"
        "extra-likelihood-kwargs = {'regularization_constant': 0.99}\n",
    )

    assert args.polarization_modes == ["pc"]
    assert args.polarization_basis == ["p"]

    interferometers = object()
    time_frequency_filter = object()
    priors = object()
    inputs = _build_inputs(
        args,
        interferometers=interferometers,
        time_frequency_filter=time_frequency_filter,
        priors=priors,
    )

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
    assert likelihood.extra_kwargs == {"regularization_constant": 0.99}


def test_extra_kwargs_dropped_when_likelihood_rejects_variadic(monkeypatch, tmp_path):
    """A likelihood without ``**kwargs`` must not receive unknown extras (no TypeError)."""
    args = _parse_config(
        tmp_path,
        "polarization-modes = pc\n"
        "polarization-basis = p\n"
        "extra-likelihood-kwargs = {'regularization_constant': 0.99}\n",
    )

    interferometers = object()
    time_frequency_filter = object()
    priors = object()
    inputs = _build_inputs(
        args,
        interferometers=interferometers,
        time_frequency_filter=time_frequency_filter,
        priors=priors,
    )

    monkeypatch.setattr(input_module, "Chi2TimeFrequencyLikelihood", _StrictLikelihood)

    # Would raise ``TypeError: unexpected keyword argument 'regularization_constant'``
    # if the unknown extra were forwarded to a signature without ``**kwargs``.
    likelihood = input_module.Input.likelihood.fget(inputs)

    assert likelihood.interferometers is interferometers
    assert likelihood.polarization_basis == "p"
    assert likelihood.time_frequency_filter is time_frequency_filter
    assert likelihood.priors is priors


def test_dotted_path_forwards_keyword_only_parameters(monkeypatch, tmp_path):
    """The ``module.Class`` likelihood_type branch also forwards keyword-only params."""
    args = _parse_config(
        tmp_path,
        "polarization-modes = pc\n"
        "polarization-basis = p\n"
        "extra-likelihood-kwargs = {'regularization_constant': 0.99}\n",
    )

    interferometers = object()
    time_frequency_filter = object()
    priors = object()
    inputs = _build_inputs(
        args,
        interferometers=interferometers,
        time_frequency_filter=time_frequency_filter,
        priors=priors,
        likelihood_type="my_pkg.CustomLikelihood",
    )

    fake_module = types.ModuleType("my_pkg")
    fake_module.CustomLikelihood = _KeywordOnlyLikelihood
    monkeypatch.setattr(input_module, "import_module", lambda name: fake_module)

    likelihood = input_module.Input.likelihood.fget(inputs)

    assert likelihood.polarization_basis == "p"
    assert likelihood.time_frequency_filter is time_frequency_filter
    assert likelihood.extra_kwargs == {"regularization_constant": 0.99}


def test_lensing_likelihood_is_rejected_by_the_cli(tmp_path):
    """The two-image library API must not enter the flat-network CLI flow."""
    args = _parse_config(tmp_path, "polarization-modes = pc\npolarization-basis = p\n")
    inputs = _build_inputs(
        args,
        interferometers=object(),
        time_frequency_filter=object(),
        priors=object(),
        likelihood_type="nullpol.analysis.lensing.chi2_tf_likelihood.LensingChi2TimeFrequencyLikelihood",
    )

    with pytest.raises(NullpolError, match="library-only likelihood"):
        input_module.Input.likelihood.fget(inputs)
