#!/usr/bin/env python

from setuptools import setup
from Cython.Distutils import Extension
from Cython.Build import cythonize
import cython_gsl
import sys
import os
import numpy as np

# Enable Cython coverage
CYTHON_COVERAGE = os.getenv("CYTHON_COVERAGE", "1") == "1"

python_version = sys.version_info

if python_version < (3, 8):
    sys.exit("Python < 3.8 is not supported, aborting setup")


def get_long_description():
    """Finds the README and reads in the description"""
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "README.rst")) as f:
        long_description = f.read()

    return long_description


def get_requirements(kind=None):
    if kind is None:
        fname = "requirements.txt"
    else:
        fname = f"{kind}_requirements.txt"
    with open(fname, "r") as ff:
        requirements = ff.readlines()

    return requirements


# get version info from __init__.py
def readfile(filename):
    with open(filename) as fp:
        filecontents = fp.read()

    return filecontents


long_description = get_long_description()

extensions  = [
    Extension("nullpol.wdm.inverse_wavelet_freq_funcs",
              ["nullpol/wdm/inverse_wavelet_freq_funcs.pyx"],
              include_dirs=[cython_gsl.get_cython_include_dir(),
                            np.get_include()],
              libraries=cython_gsl.get_libraries(),
              library_dirs=[cython_gsl.get_library_dir()],
    ),
    Extension("nullpol.wdm.inverse_wavelet_time_funcs",
              ["nullpol/wdm/inverse_wavelet_time_funcs.pyx"],
              include_dirs=[cython_gsl.get_cython_include_dir(),
                            np.get_include()],
              libraries=cython_gsl.get_libraries(),
              library_dirs=[cython_gsl.get_library_dir()],
    ),
    Extension("nullpol.wdm.transform_freq_funcs",
              ["nullpol/wdm/transform_freq_funcs.pyx"],
              include_dirs=[cython_gsl.get_cython_include_dir(),
                            np.get_include()],
              libraries=cython_gsl.get_libraries(),
              library_dirs=[cython_gsl.get_library_dir()],
    ),
    Extension("nullpol.wdm.transform_time_funcs",
              ["nullpol/wdm/transform_time_funcs.pyx"],
              include_dirs=[cython_gsl.get_cython_include_dir(),
                            np.get_include()],
              libraries=cython_gsl.get_libraries(),
              library_dirs=[cython_gsl.get_library_dir()],
    ),
    Extension("nullpol.wdm.wavelet_transform",
              ["nullpol/wdm/wavelet_transform.pyx"],
              include_dirs=[cython_gsl.get_cython_include_dir(),
                            np.get_include()],
              libraries=cython_gsl.get_libraries(),
              library_dirs=[cython_gsl.get_library_dir()],
    )
    Extension("nullpol.time_frequency_transform.inverse_wavelet_freq_funcs",
              ["nullpol/time_frequency_transform/inverse_wavelet_freq_funcs.pyx"],
              include_dirs=[cython_gsl.get_cython_include_dir(),
                            np.get_include()],
              libraries=cython_gsl.get_libraries(),
              library_dirs=[cython_gsl.get_library_dir()],
              define_macros=[("CYTHON_TRACE", "1")] if CYTHON_COVERAGE else [],
    ),
    Extension("nullpol.time_frequency_transform.inverse_wavelet_time_funcs",
              ["nullpol/time_frequency_transform/inverse_wavelet_time_funcs.pyx"],
              include_dirs=[cython_gsl.get_cython_include_dir(),
                            np.get_include()],
              libraries=cython_gsl.get_libraries(),
              library_dirs=[cython_gsl.get_library_dir()],
              define_macros=[("CYTHON_TRACE", "1")] if CYTHON_COVERAGE else [],
    ),
    Extension("nullpol.time_frequency_transform.transform_freq_funcs",
              ["nullpol/time_frequency_transform/transform_freq_funcs.pyx"],
              include_dirs=[cython_gsl.get_cython_include_dir(),
                            np.get_include()],
              libraries=cython_gsl.get_libraries(),
              library_dirs=[cython_gsl.get_library_dir()],
              define_macros=[("CYTHON_TRACE", "1")] if CYTHON_COVERAGE else [],
    ),
    Extension("nullpol.time_frequency_transform.transform_time_funcs",
              ["nullpol/time_frequency_transform/transform_time_funcs.pyx"],
              include_dirs=[cython_gsl.get_cython_include_dir(),
                            np.get_include()],
              libraries=cython_gsl.get_libraries(),
              library_dirs=[cython_gsl.get_library_dir()],
              define_macros=[("CYTHON_TRACE", "1")] if CYTHON_COVERAGE else [],
    ),
    Extension("nullpol.time_frequency_transform.wavelet_transform",
              ["nullpol/time_frequency_transform/wavelet_transform.pyx"],
              include_dirs=[cython_gsl.get_cython_include_dir(),
                            np.get_include()],
              libraries=cython_gsl.get_libraries(),
              library_dirs=[cython_gsl.get_library_dir()],
              define_macros=[("CYTHON_TRACE", "1")] if CYTHON_COVERAGE else [],
    )
]

setup(
    name="nullpol",
    description="Polarization test",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="Isaac Wong, Thomas Ng, Balázs Cirok",
    packages=[
        "nullpol",
        "nullpol.detector"
    ],
    package_dir={"nullpol": "nullpol"},
    ext_modules=cythonize(extensions, language_level="3", compiler_directives={"linetrace": CYTHON_COVERAGE}),
)

