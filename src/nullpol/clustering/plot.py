from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from gwpy.spectrogram import Spectrogram

from ..time_frequency_transform import get_shape_of_wavelet_transform


def plot_spectrogram(spectrogram,
                     duration,
                     sampling_frequency,
                     wavelet_frequency_resolution,
                     frequency_range=None,
                     t0=0,
                     title=None, savefig=None, dpi=100):
    Nt, Nf = get_shape_of_wavelet_transform(duration=duration,
                                            sampling_frequency=sampling_frequency,
                                            wavelet_frequency_resolution=wavelet_frequency_resolution)
    spectrogram = Spectrogram(spectrogram, t0=t0, dt=duration/Nt, df=wavelet_frequency_resolution, name=title)
    plot = spectrogram.plot()
    ax = plot.gca()
    if frequency_range is not None:
        ax.set_ylim(*frequency_range)
    ax.set_yscale('log')
    ax.colorbar()
    if savefig is not None:
        plt.savefig(fname=savefig, dpi=dpi, bbox_inches='tight')


def plot_reverse_cumulative_distribution(spectrogram: np.ndarray,
                                         bins: int=25,
                                         title: str | None=None,
                                         savefig: str | None=None,
                                         dpi: int=100):
    spectrogram_flatten = spectrogram.flatten()
    plt.hist(spectrogram_flatten, bins=bins, density=False, cumulative=-1, histtype='step')
    if title is not None:
        plt.title(title)
    else:
        plt.title('Reversed cumulative distribution')
    if savefig is not None:
        plt.savefig(fname=savefig, dpi=dpi, bbox_inches='tight')
