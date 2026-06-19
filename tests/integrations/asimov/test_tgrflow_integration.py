"""Integration tests for :meth:`nullpol.integrations.asimov.tgrflow.Applicator.run`.

These exercise the real :meth:`Applicator.run` code path end-to-end while
mocking only the external boundaries (``cbcflow``, ``asimov.event.Event`` and
``bilby_config_to_asimov``). They cover the behaviour changes introduced by the
``fix/bilby-asimov-compat`` branch:

* the ``event_time is None`` guard that turns a missing preferred GraceDB
  event into a clear :class:`ValueError`,
* the move from "only the IllustrativeResult, else raise" to
  :func:`identify_basis_production` (illustrative-then-fallback) selection,
* the new ``psds`` handling that distinguishes a ``None`` psd dict from an
  absent one.

The optional ``asimov`` / ``cbcflow`` packages are stubbed by the
``asimov_libs`` fixture; the stubs are then *replaced* per test with
configured mocks so the run flow can be driven deterministically.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _pe_result(uid: str = "PROD1", **overrides) -> dict:
    """Build a cbcflow-style PE result dict with host-prefixed file paths."""
    r = {
        "UID": uid,
        "RunStatus": "complete",
        "ReviewStatus": "pass",
        "WaveformApproximant": "IMRPhenomPv2",
        "ResultFile": {"Path": f"machine:/data/{uid}/result.hdf5"},
        "ConfigFile": {"Path": f"machine:/data/{uid}/config.ini"},
        "PESummaryResultFile": {"Path": f"machine:/data/{uid}/pe.h5"},
    }
    r.update(overrides)
    return r


def _fake_metadata(*, results, illustrative="PROD1", preferred=True, sname="S1234") -> SimpleNamespace:
    """Build a superevent metadata object with the fields ``Applicator.run`` reads."""
    events = [{"State": "other", "UID": "E124", "FAR": 1e-9, "GPSTime": 1234567800}]
    if preferred:
        events.insert(0, {"State": "preferred", "UID": "E123", "FAR": 1e-10, "GPSTime": 1234567890})
    pe = {"Results": results, "SafeSamplingRate": 2048}
    if illustrative is not None:
        pe["IllustrativeResult"] = illustrative
    return SimpleNamespace(
        data={
            "Sname": sname,
            "DetectorCharacterization": {
                "RecommendedDetectors": [
                    {
                        "UID": "H1",
                        "RecommendedMinimumFrequency": 20,
                        "RecommendedMaximumFrequency": 1024,
                        "RecommendedChannel": "H1:STRAIN",
                        "FrameType": "H1_TEST",
                        "FrameFile": "/data/H1.gwf",
                    },
                    {
                        "UID": "L1",
                        "RecommendedMinimumFrequency": 20,
                        "RecommendedMaximumFrequency": 1024,
                        "RecommendedChannel": "L1:STRAIN",
                        "FrameType": "L1_TEST",
                        "FrameFile": "/data/L1.gwf",
                    },
                ],
                "ParticipatingDetectors": ["H1", "L1"],
                "RecommendedDuration": 4,
            },
            "GraceDB": {"Events": events},
            "ParameterEstimation": pe,
        }
    )


def _default_bilby_config() -> dict:
    """A minimal bilby_config_to_asimov-style return value with no psds."""
    return {
        "ifos": ["H1", "L1"],
        "event time": 1234567890,
        "quality": {"sample rate": 2048},
        "data": {
            "channels": {"H1": "H1:GDCHSHIFT", "L1": "L1:GDCHSHIFT"},
            "frame types": {"H1": "H1_CFG", "L1": "L1_CFG"},
            "data files": {"H1": "/cfg/H1.gwf", "L1": "/cfg/L1.gwf"},
            "segment length": 4,
        },
        "waveform": {"approximant": "IMRPhenomPv2"},
        "likelihood": {},
        "priors": {},
    }


@pytest.fixture
def applicator_factory(asimov_libs, mocker, tmp_path):
    """Return a factory that builds a fully-wired Applicator with mocked boundaries.

    The factory returns the Applicator instance augmented with ``captured``
    (list of dicts passed to ``Event.from_dict``), ``ledger_mock``,
    ``library_mock`` and ``cbcflow_mock`` for assertions.
    """
    tgrflow = asimov_libs.tgrflow

    def _make(*, metadata, bilby_config=None, prefer_config=True):
        if bilby_config is None:
            bilby_config = _default_bilby_config()

        # --- cbcflow: get_superevent + LocalLibraryDatabase ---
        cbcflow_mock = mocker.MagicMock()
        cbcflow_mock.get_superevent.return_value = metadata
        library_mock = mocker.MagicMock()
        cbcflow_mock.core.database.LocalLibraryDatabase.return_value = library_mock
        mocker.patch.object(tgrflow, "cbcflow", cbcflow_mock)

        # --- Event.from_dict: record every dict passed and return a mock event ---
        captured = []

        def fake_from_dict(d):
            captured.append(d)
            evt = mocker.MagicMock()
            evt.work_dir = str(tmp_path)
            return evt

        event_cls = mocker.MagicMock()
        event_cls.from_dict.side_effect = fake_from_dict
        mocker.patch.object(tgrflow, "Event", event_cls)

        # --- bilby_config_to_asimov ---
        mocker.patch.object(tgrflow, "bilby_config_to_asimov", return_value=bilby_config)

        # --- ledger ---
        ledger = mocker.MagicMock()
        hook = {"library location": "/placeholder/test_library"}
        if not prefer_config:
            hook["data preference"] = "cbcflow"
        ledger.data = {"hooks": {"applicator": {"tgrflow": hook}}}

        app = tgrflow.Applicator(ledger)
        app.captured = captured
        app.ledger_mock = ledger
        app.library_mock = library_mock
        app.cbcflow_mock = cbcflow_mock
        app.event_cls = event_cls
        return app

    return _make


@pytest.mark.integration
class TestApplicatorRun:
    """Integration tests for :meth:`Applicator.run`."""

    def test_successful_run_wires_basis_result(self, applicator_factory):
        """A valid basis result is selected and its file paths are wired into gr pe info."""
        prod1 = _pe_result("PROD1")
        metadata = _fake_metadata(results=[prod1], illustrative="PROD1")
        app = applicator_factory(metadata=metadata)
        app.run("S1234")

        assert app.cbcflow_mock.get_superevent.call_args.args[0] == "S1234"
        out = app.captured[-1]
        assert out["name"] == "S1234"
        assert out["event time"] == 1234567890
        gr_pe = out["gr pe info"]
        assert gr_pe["available"] is True
        assert gr_pe["UID GR PE"] == "PROD1"
        assert gr_pe["result file path"] == "/data/PROD1/result.hdf5"
        assert gr_pe["config file path"] == "/data/PROD1/config.ini"
        assert gr_pe["pesummary result path"] == "/data/PROD1/pe.h5"
        assert gr_pe["approximant"] == "IMRPhenomPv2"
        assert out["quality"]["waveform approximant"] == "IMRPhenomPv2"
        app.ledger_mock.add_event.assert_called_once()
        app.ledger_mock.update_event.assert_called_once()

    def test_missing_preferred_event_raises(self, applicator_factory):
        """No preferred GraceDB event raises ValueError (the new guard)."""
        metadata = _fake_metadata(results=[_pe_result()], preferred=False)
        app = applicator_factory(metadata=metadata)
        with pytest.raises(ValueError, match="No preferred GraceDB event"):
            app.run("S1234")
        # Nothing should have been written to the ledger.
        app.ledger_mock.add_event.assert_not_called()
        app.ledger_mock.update_event.assert_not_called()

    def test_no_valid_basis_result_raises(self, applicator_factory):
        """When no PE result passes validation, AttributeError is raised."""
        metadata = _fake_metadata(
            results=[
                _pe_result("online"),
                _pe_result("PROD1", RunStatus="running"),
                _pe_result("PROD2", ReviewStatus="fail"),
            ],
            illustrative="PROD1",
        )
        app = applicator_factory(metadata=metadata)
        with pytest.raises(AttributeError, match="No valid GR PE result"):
            app.run("S1234")
        app.ledger_mock.add_event.assert_not_called()
        app.ledger_mock.update_event.assert_not_called()

    def test_falls_back_from_invalid_illustrative(self, applicator_factory):
        """An invalid IllustrativeResult falls back to another valid result."""
        illustrative = _pe_result("PROD1", RunStatus="running")
        fallback = _pe_result("PROD2")
        metadata = _fake_metadata(results=[illustrative, fallback], illustrative="PROD1")
        app = applicator_factory(metadata=metadata)
        app.run("S1234")
        assert app.captured[-1]["gr pe info"]["UID GR PE"] == "PROD2"

    def test_psds_copied_to_psd_folder(self, applicator_factory, tmp_path):
        """When bilby_config has a psds dict, files are copied under PSDs/ and wired in."""
        src_h1 = tmp_path / "H1_src.dat"
        src_l1 = tmp_path / "L1_src.dat"
        src_h1.write_text("psd H1")
        src_l1.write_text("psd L1")

        bilby_config = _default_bilby_config()
        bilby_config["psds"] = {"H1": str(src_h1), "L1": str(src_l1)}
        metadata = _fake_metadata(results=[_pe_result()])
        app = applicator_factory(metadata=metadata, bilby_config=bilby_config)
        app.run("S1234")

        out = app.captured[-1]
        # gr pe info keeps the original source paths
        assert out["gr pe info"]["psds"] == {"H1": str(src_h1), "L1": str(src_l1)}
        # output["psds"] points at the copied files inside PSDs/
        assert "psds" in out
        assert out["psds"]["H1"].endswith("PSDs/H1_psd.dat")
        assert out["psds"]["L1"].endswith("PSDs/L1_psd.dat")
        # and the copied files physically exist
        import os

        assert os.path.exists(out["psds"]["H1"])
        assert os.path.exists(out["psds"]["L1"])

    def test_psds_none_not_copied(self, applicator_factory):
        """A ``None`` psds dict is popped and no PSD copy / output entry is produced."""
        bilby_config = _default_bilby_config()
        bilby_config["psds"] = None
        metadata = _fake_metadata(results=[_pe_result()])
        app = applicator_factory(metadata=metadata, bilby_config=bilby_config)
        app.run("S1234")

        out = app.captured[-1]
        assert "psds" not in out
        assert "psds" not in out["gr pe info"]
