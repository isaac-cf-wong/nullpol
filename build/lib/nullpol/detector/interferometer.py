from bilby.gw.detector.interferometer import (
    Interferometer as BilbyInterferometer
)
from bilby.gw.detector.calibration import Recalibrate


class Interferometer(BilbyInterferometer):
    """A class to represent a single interferometer
    """
    def __init__(self, name, power_spectral_density, minimum_frequency, maximum_frequency, length, latitude, longitude,
                 elevation, xarm_azimuth, yarm_azimuth, xarm_tilt=0., yarm_tilt=0., calibration_model=Recalibrate()):
        super(Interferometer, self).__init__(name=name, power_spectral_density=power_spectral_density,
                                             minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency,
                                             length=length, latitude=latitude, longitude=longitude,
                                             elevation=elevation, xarm_azimuth=xarm_azimuth,
                                             yarm_azimuth=yarm_azimuth, xarm_tilt=xarm_tilt, yarm_tilt=yarm_tilt,
                                             calibration_model=calibration_model)
