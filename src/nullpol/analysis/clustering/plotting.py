from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from gwpy.spectrogram import Spectrogram

from ..tf_transforms import get_shape_of_wavelet_transform


def plot_spectrogram(
    spectrogram,
    duration,
    sampling_frequency,
    wavelet_frequency_resolution,
    frequency_range=None,
    t0=0,
    title=None,
    savefig=None,
    dpi=100,
):
    """Plot a time-frequency spectrogram.

    Creates a visual representation of the time-frequency spectrogram data
    using GWpy's plotting capabilities with logarithmic frequency scaling.

    Args:
        spectrogram (numpy.ndarray): 2D spectrogram data with shape (n_time, n_frequencies).
        duration (float): Total duration of the data in seconds.
        sampling_frequency (float): Sampling frequency in Hz.
        wavelet_frequency_resolution (float): Frequency resolution of the wavelet transform in Hz.
        frequency_range (tuple[float, float], optional): Frequency range (f_min, f_max) for y-axis limits.
            If None, uses the full frequency range.
        t0 (float, optional): Start time for the plot in seconds. Defaults to 0.
        title (str, optional): Title for the plot. Defaults to None.
        savefig (str, optional): Filename to save the plot. If None, plot is not saved.
        dpi (int, optional): DPI for saved figure. Defaults to 100.
    """
    Nt, Nf = get_shape_of_wavelet_transform(
        duration=duration,
        sampling_frequency=sampling_frequency,
        wavelet_frequency_resolution=wavelet_frequency_resolution,
    )
    spectrogram = Spectrogram(spectrogram, t0=t0, dt=duration / Nt, df=wavelet_frequency_resolution, name=title)
    plot = spectrogram.plot()
    ax = plot.gca()
    if frequency_range is not None:
        ax.set_ylim(*frequency_range)
    ax.set_yscale("log")
    ax.colorbar()
    if savefig is not None:
        plt.savefig(fname=savefig, dpi=dpi, bbox_inches="tight")


def plot_reverse_cumulative_distribution(
    spectrogram: np.ndarray, bins: int = 25, title: str | None = None, savefig: str | None = None, dpi: int = 100
):
    """Plot reverse cumulative distribution of spectrogram values.

    Creates a histogram showing the reverse cumulative distribution of values
    in the spectrogram, useful for understanding the statistical distribution
    of power and selecting appropriate thresholds.

    Args:
        spectrogram (numpy.ndarray): 2D spectrogram data to analyze.
        bins (int, optional): Number of histogram bins. Defaults to 25.
        title (str, optional): Title for the plot. If None, uses default title.
        savefig (str, optional): Filename to save the plot. If None, plot is not saved.
        dpi (int, optional): DPI for saved figure. Defaults to 100.
    """
    spectrogram_flatten = spectrogram.flatten()
    plt.hist(spectrogram_flatten, bins=bins, density=False, cumulative=-1, histtype="step")
    if title is not None:
        plt.title(title)
    else:
        plt.title("Reversed cumulative distribution")
    if savefig is not None:
        plt.savefig(fname=savefig, dpi=dpi, bbox_inches="tight")
