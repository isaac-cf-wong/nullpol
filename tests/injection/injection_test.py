"""Test module for signal injection functionality.

This module tests the injection of simulated gravitational wave signals into
detector data streams.
"""

from __future__ import annotations

import unittest

from bilby.gw.detector import InterferometerList
from bilby.gw.source import lal_binary_black_hole

from nullpol.injection import create_injection


class TestInjection(unittest.TestCase):
    """Test class for signal injection procedures.

    This class validates the injection of binary black hole signals into
    detector strain data under various noise conditions, ensuring proper
    signal generation and detector response calculation.
    """

    def test_create_injection_zero_noise(self):
        """Test signal injection into noiseless detector data.

        Validates that binary black hole signals can be properly injected
        into zero-noise detector data, testing the fundamental signal
        generation and detector response computation without noise contamination.
        """
        interferometers = InterferometerList(["H1", "L1", "V1"])
        parameters = dict(
            mass_1=36.0,
            mass_2=29.0,
            a_1=0.4,
            a_2=0.3,
            tilt_1=0.5,
            tilt_2=1.0,
            phi_12=1.7,
            phi_jl=0.3,
            luminosity_distance=2000.0,
            theta_jn=0.4,
            psi=2.659,
            phase=1.3,
            geocent_time=1126259642.413,
            ra=1.375,
            dec=-1.2108,
        )
        duration = 8
        sampling_frequency = 4096
        start_time = parameters["geocent_time"] - 4
        noise_type = "zero_noise"
        freuency_domain_source_model = lal_binary_black_hole
        waveform_arguments = dict(waveform_approximant="IMRPhenomPv2", reference_frequency=50)
        create_injection(
            interferometers=interferometers,
            parameters=parameters,
            duration=duration,
            sampling_frequency=sampling_frequency,
            start_time=start_time,
            noise_type=noise_type,
            frequency_domain_source_model=freuency_domain_source_model,
            waveform_arguments=waveform_arguments,
        )

    def test_create_injection_gaussian(self):
        """Test signal injection into Gaussian noise.

        Validates that binary black hole signals can be properly injected
        into detector data with simulated Gaussian noise, testing realistic
        signal-plus-noise scenarios for detection algorithm validation.
        """
        interferometers = InterferometerList(["H1", "L1", "V1"])
        parameters = dict(
            mass_1=36.0,
            mass_2=29.0,
            a_1=0.4,
            a_2=0.3,
            tilt_1=0.5,
            tilt_2=1.0,
            phi_12=1.7,
            phi_jl=0.3,
            luminosity_distance=2000.0,
            theta_jn=0.4,
            psi=2.659,
            phase=1.3,
            geocent_time=1126259642.413,
            ra=1.375,
            dec=-1.2108,
        )
        duration = 8
        sampling_frequency = 4096
        start_time = parameters["geocent_time"] - 4
        noise_type = "gaussian"
        freuency_domain_source_model = lal_binary_black_hole
        waveform_arguments = dict(waveform_approximant="IMRPhenomPv2", reference_frequency=50)
        create_injection(
            interferometers=interferometers,
            parameters=parameters,
            duration=duration,
            sampling_frequency=sampling_frequency,
            start_time=start_time,
            noise_type=noise_type,
            frequency_domain_source_model=freuency_domain_source_model,
            waveform_arguments=waveform_arguments,
        )

    def test_create_injection_noise(self):
        """Test noise-only injection for background characterization.

        Validates the generation of noise-only detector data without
        gravitational wave signals, essential for background characterization
        and noise property studies.
        """
        interferometers = InterferometerList(["H1", "L1", "V1"])
        duration = 8
        sampling_frequency = 4096
        start_time = 0
        noise_type = "noise"
        create_injection(
            interferometers=interferometers,
            duration=duration,
            sampling_frequency=sampling_frequency,
            start_time=start_time,
            noise_type=noise_type,
        )


if __name__ == "__main__":
    unittest.main()
