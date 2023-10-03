#!/usr/bin/env python

from setuptools import setup
import sys
import os

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

setup(
    name="nullpol",
    description="Polarization test",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="Isaac Wong, Thomas Ng",
    license="MIT",
    packages=[
        "nullpol",
        "nullpol.detector"
    ],
    package_dir={"nullpol": "nullpol"},
    python_requires=">=3.8",
    install_requires=get_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

