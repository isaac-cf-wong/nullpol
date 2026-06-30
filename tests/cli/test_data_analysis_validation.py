"""Unit tests for :meth:`DataAnalysisInput._validate_polarization_model_setting`.

This guards the fix in commit ``74b4fc3`` where the supported-modes list
contained a duplicate ``"b"`` and was missing ``"p"`` (plus), causing a valid
``"p"`` mode to be rejected in both ``polarization-modes`` and
``polarization-basis`` validation.
"""

from __future__ import annotations

import pytest

from nullpol.cli.data_analysis import DataAnalysisInput
from nullpol.utils import NullpolError

SUPPORTED = ["b", "c", "p", "l", "x", "y"]


def _make_input(modes, basis):
    """Build a DataAnalysisInput bypassing the heavy __init__.

    The validation method only reads the ``polarization_modes`` and
    ``polarization_basis`` properties, which are backed by the private
    ``_polarization_modes`` / ``_polarization_basis`` attributes.
    """
    obj = object.__new__(DataAnalysisInput)
    obj._polarization_modes = list(modes)
    obj._polarization_basis = list(basis)
    return obj


class TestValidatePolarizationModelSetting:
    """Tests for :meth:`_validate_polarization_model_setting`."""

    @pytest.mark.parametrize("mode", SUPPORTED)
    def test_each_supported_mode_accepted_alone(self, mode):
        """Every one of the six canonical modes is accepted in polarization-modes."""
        _make_input([mode], [mode])._validate_polarization_model_setting()

    def test_all_six_modes_accepted_together(self):
        """All six modes (b, c, p, l, x, y) pass when used together."""
        _make_input(SUPPORTED, SUPPORTED)._validate_polarization_model_setting()

    def test_plus_mode_accepted(self):
        """Regression: ``"p"`` (plus) must be accepted -- it was rejected before the fix."""
        _make_input(["p", "c"], ["p"])._validate_polarization_model_setting()

    def test_plus_mode_rejected_before_fix_list(self):
        """Sanity check: the fixed list no longer contains a duplicate ``b``."""
        # If the old buggy list ["b","c","b","l","x","y"] were restored, "p" would fail.
        assert SUPPORTED == ["b", "c", "p", "l", "x", "y"]
        assert SUPPORTED.count("b") == 1
        assert "p" in SUPPORTED

    @pytest.mark.parametrize("bad_mode", ["z", "a", "B", "pc", "", "plus"])
    def test_invalid_mode_in_modes_rejected(self, bad_mode):
        """An unsupported mode in ``polarization-modes`` raises NullpolError."""
        with pytest.raises(NullpolError, match="polarization-modes"):
            _make_input([bad_mode], [bad_mode])._validate_polarization_model_setting()

    @pytest.mark.parametrize("bad_mode", ["z", "a", "P"])
    def test_invalid_mode_in_basis_rejected(self, bad_mode):
        """An unsupported mode in ``polarization-basis`` raises NullpolError."""
        with pytest.raises(NullpolError, match="polarization-basis"):
            _make_input(["p", "c"], [bad_mode])._validate_polarization_model_setting()

    def test_basis_not_subset_of_modes_rejected(self):
        """A basis mode that is supported but not in ``polarization-modes`` is rejected."""
        # "b" is a supported mode, but it is not in the modes list here.
        with pytest.raises(NullpolError, match="Basis mode"):
            _make_input(["p", "c"], ["b"])._validate_polarization_model_setting()

    def test_empty_basis_accepted(self):
        """An empty basis is valid (all modes are derived)."""
        _make_input(["p", "c"], [])._validate_polarization_model_setting()

    def test_basis_subset_of_modes_accepted(self):
        """A basis that is a strict subset of the modes is valid."""
        _make_input(["p", "c", "b", "l"], ["p", "c"])._validate_polarization_model_setting()
