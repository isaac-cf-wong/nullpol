from __future__ import annotations

import numpy as np

POLARIZATION_ENCODING = dict(p=0,c=1,b=2,l=3,x=4,y=5)
POLARIZATION_DECODING = np.array(['p', 'c', 'b', 'l', 'x', 'y'])
POLARIZATION_LONG_NAMES = dict(p='plus', c='cross', b='breathing', l='longitudinal', x='vector_x', y='vector_y')
POLARIZATION_SHORT_NAMES = dict(plus='p', cross='c', breathing='b', longitudinal='l', vector_x='x', vector_y='y')


def encode_polarization(polarization_modes, polarization_basis):
    _polarization_modes = np.full(6, False)
    _polarization_basis = np.full(6, False)
    for pol in polarization_modes:
        _polarization_modes[POLARIZATION_ENCODING[pol]] = True
    for pol in polarization_basis:
        _polarization_basis[POLARIZATION_ENCODING[pol]] = True
    _polarization_derived = _polarization_modes & (~_polarization_basis)
    return _polarization_modes, _polarization_basis, _polarization_derived


def get_long_names(tokens):
    return [POLARIZATION_LONG_NAMES[token] for token in tokens]
