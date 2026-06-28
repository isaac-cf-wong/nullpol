"""Unit tests for :mod:`nullpol.integrations.asimov.tgrflow`.

These cover the pure-Python helpers introduced / hardened in the
``fix/bilby-asimov-compat`` branch:

* :func:`validate_gr_pe_result` -- decides whether a single PE result is
  suitable as the basis for a TGR analysis.
* :func:`identify_basis_production` -- selects the basis PE result, preferring
  the ``IllustrativeResult`` and falling back to the first valid result.
* :meth:`Collector._get_pe_result_from_production` -- the ``review is None``
  and missing-``Notes`` defensive guards.

The optional ``asimov`` / ``cbcflow`` dependencies are stubbed out by the
``asimov_libs`` fixture in ``conftest.py`` so the tests run in the default
environment.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest


def _valid_result(uid: str = "PROD1", **overrides) -> dict:
    """Build a PE result dict that passes :func:`validate_gr_pe_result`."""
    result = {
        "UID": uid,
        "RunStatus": "complete",
        "ReviewStatus": "pass",
        "ResultFile": {"Path": "host:/data/result.hdf5"},
        "ConfigFile": {"Path": "host:/data/config.ini"},
        "PESummaryResultFile": {"Path": "host:/data/pe.h5"},
    }
    result.update(overrides)
    return result


def _metadata(results: list, illustrative: str | None = None) -> SimpleNamespace:
    """Build a minimal cbcflow-like superevent metadata object."""
    pe = {"Results": results}
    if illustrative is not None:
        pe["IllustrativeResult"] = illustrative
    return SimpleNamespace(data={"ParameterEstimation": pe})


# ---------------------------------------------------------------------------
# validate_gr_pe_result
# ---------------------------------------------------------------------------


class TestValidateGrPeResult:
    """Tests for :func:`validate_gr_pe_result`."""

    def test_valid_result(self, asimov_libs):
        """A complete, approved, fully-filed result is valid."""
        assert asimov_libs.validate_gr_pe_result(_valid_result()) is True

    def test_extra_fields_ignored(self, asimov_libs):
        """Extra metadata fields (e.g. WaveformApproximant) do not invalidate."""
        assert asimov_libs.validate_gr_pe_result(_valid_result(WaveformApproximant="IMRPhenomPv2")) is True

    @pytest.mark.parametrize("uid", ["online", "Online", "ONLINE"])
    def test_online_uid_rejected(self, asimov_libs, uid):
        """``online`` runs are never suitable as a TGR basis (case-insensitive)."""
        assert asimov_libs.validate_gr_pe_result(_valid_result(uid=uid)) is False

    @pytest.mark.parametrize("uid", ["exp1", "EXP_test", "exploratory"])
    def test_exploratory_uid_rejected(self, asimov_libs, uid):
        """UIDs starting with ``exp`` are excluded."""
        assert asimov_libs.validate_gr_pe_result(_valid_result(uid=uid)) is False

    @pytest.mark.parametrize("uid", ["detchar_xyz", "DETCHAR_1"])
    def test_detchar_uid_rejected(self, asimov_libs, uid):
        """UIDs starting with ``detchar`` are excluded."""
        assert asimov_libs.validate_gr_pe_result(_valid_result(uid=uid)) is False

    def test_incomplete_run_status_rejected(self, asimov_libs):
        """Results that are not ``complete`` are rejected."""
        assert asimov_libs.validate_gr_pe_result(_valid_result(RunStatus="running")) is False

    def test_unapproved_review_rejected(self, asimov_libs):
        """Only ``pass`` review status is accepted."""
        assert asimov_libs.validate_gr_pe_result(_valid_result(ReviewStatus="fail")) is False

    def test_missing_review_rejected(self, asimov_libs):
        """A result with no review status is rejected."""
        result = _valid_result()
        result.pop("ReviewStatus")
        assert asimov_libs.validate_gr_pe_result(result) is False

    def test_deprecated_rejected(self, asimov_libs):
        """Deprecated results are rejected even if otherwise complete."""
        assert asimov_libs.validate_gr_pe_result(_valid_result(Deprecated=True)) is False

    @pytest.mark.parametrize("missing", ["ResultFile", "ConfigFile", "PESummaryResultFile"])
    def test_missing_required_file_rejected(self, asimov_libs, missing):
        """A result missing any required file is rejected."""
        result = _valid_result()
        result.pop(missing)
        assert asimov_libs.validate_gr_pe_result(result) is False

    def test_normal_uid_with_exp_prefix_only_in_middle(self, asimov_libs):
        """``exp`` / ``detchar`` must be a *prefix*, not a substring."""
        # "exp" is not a prefix of "PROD_exp" -> should not be rejected on UID grounds.
        assert asimov_libs.validate_gr_pe_result(_valid_result(uid="PROD_exp")) is True


# ---------------------------------------------------------------------------
# identify_basis_production
# ---------------------------------------------------------------------------


class TestIdentifyBasisProduction:
    """Tests for :func:`identify_basis_production`."""

    def test_illustrative_returned_when_valid(self, asimov_libs):
        """The IllustrativeResult is returned when it passes validation."""
        prod1 = _valid_result("PROD1")
        prod2 = _valid_result("PROD2")
        metadata = _metadata([prod1, prod2], illustrative="PROD2")
        assert asimov_libs.identify_basis_production(metadata) is prod2

    def test_illustrative_invalid_falls_back(self, asimov_libs):
        """If the IllustrativeResult fails validation, fall back to first valid."""
        illustrative = _valid_result("PROD1", RunStatus="running")
        valid = _valid_result("PROD2")
        metadata = _metadata([illustrative, valid], illustrative="PROD1")
        assert asimov_libs.identify_basis_production(metadata) is valid

    def test_illustrative_not_in_results_falls_back(self, asimov_libs):
        """A stale IllustrativeResult name falls back to the first valid result."""
        valid = _valid_result("PROD2")
        metadata = _metadata([valid], illustrative="PROD_GONE")
        assert asimov_libs.identify_basis_production(metadata) is valid

    def test_no_illustrative_returns_first_valid(self, asimov_libs):
        """Without an IllustrativeResult, the first valid result is returned."""
        prod1 = _valid_result("PROD1")
        prod2 = _valid_result("PROD2")
        metadata = _metadata([prod1, prod2])
        assert asimov_libs.identify_basis_production(metadata) is prod1

    def test_skips_invalid_results(self, asimov_libs):
        """Invalid results before the valid one are skipped."""
        invalid1 = _valid_result("online")
        invalid2 = _valid_result("PROD1", RunStatus="running")
        invalid3 = _valid_result("PROD2", Deprecated=True)
        valid = _valid_result("PROD3")
        metadata = _metadata([invalid1, invalid2, invalid3, valid])
        assert asimov_libs.identify_basis_production(metadata) is valid

    def test_no_valid_results_returns_empty(self, asimov_libs):
        """An empty dict is returned when no result passes validation."""
        metadata = _metadata([_valid_result("online"), _valid_result("PROD1", RunStatus="running")])
        assert asimov_libs.identify_basis_production(metadata) == {}

    def test_empty_results_returns_empty(self, asimov_libs):
        """An empty results list yields an empty dict."""
        metadata = _metadata([], illustrative="PROD1")
        assert asimov_libs.identify_basis_production(metadata) == {}

    def test_no_illustrative_key_in_metadata(self, asimov_libs):
        """Missing ``IllustrativeResult`` key (no .get default) is tolerated."""
        valid = _valid_result("PROD1")
        metadata = SimpleNamespace(data={"ParameterEstimation": {"Results": [valid]}})
        assert asimov_libs.identify_basis_production(metadata) is valid

    def test_illustrative_chosen_over_earlier_valid(self, asimov_libs):
        """The illustrative result wins even if an earlier valid result exists."""
        earlier = _valid_result("PROD1")
        illustrative = _valid_result("PROD2")
        metadata = _metadata([earlier, illustrative], illustrative="PROD2")
        assert asimov_libs.identify_basis_production(metadata) is illustrative


# ---------------------------------------------------------------------------
# Collector._get_pe_result_from_production -- review / Notes guards
# ---------------------------------------------------------------------------


def _mock_analysis(*, comment=None, review=None, status="finished", finished=False):
    """Build a mock asimov production with the attributes accessed by the collector."""
    analysis = mock.MagicMock()
    analysis.name = "PROD1"
    analysis.status = status
    analysis.meta = {}
    analysis.comment = comment
    analysis.review = review
    analysis.finished = finished
    # str(analysis.pipeline) is used for InferenceSoftware
    analysis.pipeline.__str__ = mock.Mock(return_value="nullpol")
    # find_prods returns a MagicMock; indexing [0] yields a MagicMock (no IndexError)
    analysis.pipeline.production.event.repository.find_prods.return_value = ["/path/to/config.ini"]
    return analysis


class TestCollectorReviewGuards:
    """Tests for the defensive guards added to ``_get_pe_result_from_production``."""

    def _collector(self, asimov_libs):
        """Create a Collector instance without running its heavy __init__."""
        collector = object.__new__(asimov_libs.Collector)
        return collector

    def test_review_none_does_not_raise(self, asimov_libs):
        """``analysis.review is None`` must not raise (was AttributeError before)."""
        collector = self._collector(asimov_libs)
        analysis = _mock_analysis(review=None, comment=None)
        # corresponding_analysis is None -> only the comment branch is skipped.
        out = collector._get_pe_result_from_production(analysis, None)
        assert out["UID"] == "PROD1"
        assert out["Notes"] == []

    def test_comment_added_when_corresponding_missing_notes(self, asimov_libs):
        """A missing ``Notes`` key on the corresponding analysis uses .get default."""
        collector = self._collector(asimov_libs)
        analysis = _mock_analysis(review=None, comment="looks good")
        out = collector._get_pe_result_from_production(analysis, {})
        assert out["Notes"] == ["looks good"]

    def test_comment_not_duplicated_when_present(self, asimov_libs):
        """An existing comment in Notes is not appended again."""
        collector = self._collector(asimov_libs)
        analysis = _mock_analysis(review=None, comment="looks good")
        out = collector._get_pe_result_from_production(analysis, {"Notes": ["looks good"]})
        assert out["Notes"] == []

    def test_review_messages_with_corresponding_missing_notes(self, asimov_libs):
        """A missing ``Notes`` key on the corresponding analysis no longer raises.

        Before the fix, ``corresponding_analysis["Notes"]`` raised ``KeyError``;
        the ``.get("Notes", [])`` fallback now evaluates cleanly. The message is
        appended because the note does not yet appear in the (empty) Notes list.
        """
        collector = self._collector(asimov_libs)

        msg = mock.MagicMock()
        msg.timestamp = __import__("datetime").datetime(2024, 1, 2, 3, 4, 5)
        msg.message = "approved"

        review = mock.MagicMock()
        review.status = "approved"
        review.messages = [msg]
        analysis = _mock_analysis(review=review, comment=None)

        # corresponding_analysis lacks a "Notes" key -- this is the crash scenario
        out = collector._get_pe_result_from_production(analysis, {})
        assert out["ReviewStatus"] == "pass"
        assert out["Notes"] == ["2024-01-02: approved"]

    def test_review_message_appended_when_corresponding_none(self, asimov_libs):
        """When there is no corresponding analysis, the review message is appended."""
        collector = self._collector(asimov_libs)

        msg = mock.MagicMock()
        msg.timestamp = __import__("datetime").datetime(2024, 1, 2, 3, 4, 5)
        msg.message = "approved"

        review = mock.MagicMock()
        review.status = "approved"
        review.messages = [msg]
        analysis = _mock_analysis(review=review, comment=None)

        out = collector._get_pe_result_from_production(analysis, None)
        assert out["ReviewStatus"] == "pass"
        assert out["Notes"] == ["2024-01-02: approved"]

    def test_review_rejected_sets_fail(self, asimov_libs):
        """A ``rejected`` review maps to ``ReviewStatus = fail``."""
        collector = self._collector(asimov_libs)
        review = mock.MagicMock()
        review.status = "rejected"
        review.messages = []
        analysis = _mock_analysis(review=review, comment=None)
        out = collector._get_pe_result_from_production(analysis, None)
        assert out["ReviewStatus"] == "fail"

    def test_review_deprecated_sets_flag(self, asimov_libs):
        """A ``deprecated`` review sets the ``Deprecated`` flag."""
        collector = self._collector(asimov_libs)
        review = mock.MagicMock()
        review.status = "deprecated"
        review.messages = []
        analysis = _mock_analysis(review=review, comment=None)
        out = collector._get_pe_result_from_production(analysis, None)
        assert out["Deprecated"] is True
