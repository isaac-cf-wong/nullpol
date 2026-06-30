"""Unit tests for :mod:`nullpol.integrations.asimov.utility`.

Focuses on :func:`_convert_string_to_dict`, which was extended in the
``fix/bilby-asimov-compat`` branch to:

* also catch :class:`TypeError` from newer ``bilby_pipe`` versions, and
* fall back to :func:`ast.literal_eval` before returning the raw string.

The optional ``asimov`` / ``cbcflow`` dependencies are stubbed out by the
``asimov_libs`` fixture in ``conftest.py``; ``bilby_pipe`` itself is a hard
dependency and is exercised for real.
"""

from __future__ import annotations

import pytest


class TestConvertStringToDict:
    """Tests for :func:`utility._convert_string_to_dict`."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ('{"H1": 1, "L1": 2}', {"H1": 1, "L1": 2}),
            ("{'H1': 1}", {"H1": 1}),
        ],
    )
    def test_json_dict_strings_parsed(self, asimov_libs, raw, expected):
        """Standard JSON / Python-literal dict strings are parsed by bilby_pipe."""
        assert asimov_libs.convert_string_to_dict(raw) == expected

    def test_bilby_pipe_style_dict_parsed(self, asimov_libs):
        """bilby_pipe's own dict syntax (unquoted keys) is parsed by bilby_pipe."""
        assert asimov_libs.convert_string_to_dict("{H1: a, L1: b}") == {"H1": "a", "L1": "b"}

    @pytest.mark.parametrize(("raw", "expected"), [("20", 20), ("1.5", 1.5), ("0", 0), ("1e5", 100000.0)])
    def test_numeric_string_parsed_via_literal_eval(self, asimov_libs, raw, expected):
        """Numeric strings that bilby_pipe rejects are now parsed by literal_eval.

        This is the behaviour change introduced by the fallback: previously a
        bare numeric string was returned unchanged (a ``str``); it is now
        returned as the corresponding numeric type.
        """
        result = asimov_libs.convert_string_to_dict(raw)
        assert result == expected
        assert isinstance(result, type(expected))

    def test_none_string_returns_none(self, asimov_libs):
        """The string ``'None'`` parses to ``None`` via both code paths."""
        assert asimov_libs.convert_string_to_dict("None") is None

    @pytest.mark.parametrize("raw", ["H1:L1", "a-b-c", "just_a_word", "true"])
    def test_unparsable_strings_returned_as_is(self, asimov_libs, raw):
        """Strings that neither bilby_pipe nor literal_eval can parse are returned unchanged."""
        result = asimov_libs.convert_string_to_dict(raw)
        assert result == raw
        assert isinstance(result, str)

    def test_empty_string_returned_as_is(self, asimov_libs):
        """An empty string cannot be parsed and is returned unchanged."""
        result = asimov_libs.convert_string_to_dict("")
        assert result == ""

    def test_none_input_returned_as_none(self, asimov_libs):
        """A ``None`` input does not raise and returns ``None``."""
        assert asimov_libs.convert_string_to_dict(None) is None


class TestDeepUpdate:
    """Sanity tests for :func:`utility.deep_update` used by the applicator."""

    def test_nested_merge(self, asimov_libs):
        """Nested dicts are merged recursively rather than overwritten."""
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        update = {"a": {"c": 20, "e": 30}}
        out = asimov_libs.deep_update(base, update)
        assert out == {"a": {"b": 1, "c": 20, "e": 30}, "d": 3}
        # original is not mutated
        assert base == {"a": {"b": 1, "c": 2}, "d": 3}

    def test_scalar_overwrites_dict(self, asimov_libs):
        """A scalar update value replaces a dict value outright."""
        out = asimov_libs.deep_update({"a": {"b": 1}}, {"a": 5})
        assert out == {"a": 5}
