import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from nullpol.tools.data_generation import DataGenerationInput


class FakeInterferometer:
    def __init__(self, name):
        self.name = name
        self.power_spectral_density = object()
        self.frequency_domain_strain = np.array([0.0, 0.0])
        self.duration = 1.0
        self.sampling_frequency = 2.0


class FakeInterferometerList(list):
    injected_waveform_generator = None
    injected_parameters = None

    def __init__(self, names):
        super().__init__(FakeInterferometer(name) for name in names)

    def set_strain_data_from_zero_noise(self, **kwargs):
        pass

    def inject_signal(self, waveform_generator, parameters, raise_error):
        self.__class__.injected_waveform_generator = waveform_generator
        self.__class__.injected_parameters = parameters


class FakeWaveformGenerator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def fake_source_model(*args, **kwargs):
    return None


def fake_parameter_conversion(*args, **kwargs):
    return None


def make_data_generation_input(directory):
    data = DataGenerationInput.__new__(DataGenerationInput)
    data.time_frequency_clustering_method = "maxL"
    data.time_frequency_filter_file = None
    data.time_frequency_clustering_pe_samples_filename = "pe_result.hdf5"
    data.time_frequency_clustering_threshold = 1.0
    data.time_frequency_clustering_threshold_type = "variance"
    data.time_frequency_clustering_time_padding = 0.1
    data.time_frequency_clustering_frequency_padding = 1.0
    data.time_frequency_clustering_skypoints = 1
    data._detectors = ["H1", "L1"]
    data._interferometers = FakeInterferometerList(["H1", "L1"])
    data.sampling_frequency = 2.0
    data.duration = 1.0
    data.start_time = 0.0
    data.enforce_signal_duration = False
    data.wavelet_frequency_resolution = 1.0
    data.wavelet_nx = 4.0
    data.minimum_frequency = 0.0
    data.maximum_frequency = 1.0
    data._outdir = str(directory)
    data.label = "test"
    data.meta_data = {}
    return data


def test_pe_clustering_uses_waveform_metadata_from_result():
    posterior = pd.DataFrame(
        [
            dict(log_likelihood=1.0, mass_1=20.0),
            dict(log_likelihood=2.0, mass_1=30.0),
        ]
    )
    pe_result = SimpleNamespace(
        posterior=posterior,
        meta_data=dict(
            likelihood=dict(
                waveform_generator_class=FakeWaveformGenerator,
                frequency_domain_source_model=fake_source_model,
                parameter_conversion=fake_parameter_conversion,
                waveform_arguments=dict(waveform_approximant="PEWaveform"),
            )
        ),
    )
    with tempfile.TemporaryDirectory() as directory:
        data = make_data_generation_input(directory)
        with mock.patch(
            "nullpol.tools.data_generation.bilby.core.result.read_in_result",
            return_value=pe_result,
        ), mock.patch(
            "nullpol.tools.data_generation.bilby.gw.detector.InterferometerList",
            FakeInterferometerList,
        ), mock.patch(
            "nullpol.tools.data_generation._run_time_frequency_clustering",
            return_value=(np.ones((2, 2)), np.ones((2, 2))),
        ), mock.patch(
            "nullpol.tools.data_generation.compute_sky_maximized_spectrogram",
            return_value=np.ones((2, 2)),
        ), mock.patch("nullpol.tools.data_generation.plot_spectrogram"), mock.patch(
            "nullpol.tools.data_generation.plot_reverse_cumulative_distribution"
        ):
            data.run_time_frequency_clustering()

        waveform_generator = FakeInterferometerList.injected_waveform_generator
        assert isinstance(waveform_generator, FakeWaveformGenerator)
        assert waveform_generator.kwargs["frequency_domain_source_model"] is fake_source_model
        assert waveform_generator.kwargs["parameter_conversion"] is fake_parameter_conversion
        assert waveform_generator.kwargs["waveform_arguments"] == {"waveform_approximant": "PEWaveform"}
        assert FakeInterferometerList.injected_parameters["mass_1"] == 30.0
        assert np.array_equal(
            np.load(Path(directory) / "data" / "test_time_frequency_filter.npy", allow_pickle=False),
            np.ones((2, 2)),
        )


def test_filter_file_is_loaded():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        source_filter = directory / "source_filter.npy"
        expected_filter = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.save(source_filter, expected_filter, allow_pickle=False)
        data = make_data_generation_input(directory)
        data.time_frequency_filter_file = str(source_filter)
        with mock.patch(
            "nullpol.tools.data_generation.compute_sky_maximized_spectrogram",
            return_value=np.ones((2, 2)),
        ), mock.patch("nullpol.tools.data_generation.plot_spectrogram"):
            data.run_time_frequency_clustering()

        assert np.array_equal(data.meta_data["time_frequency_filter"], expected_filter)
