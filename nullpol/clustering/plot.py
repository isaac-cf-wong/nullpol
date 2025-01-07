import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import librosa.display
import numpy as np
from ..time_frequency_transform import get_shape_of_wavelet_transform


def plot_spectrogram(spectrogram,
                     duration,
                     sampling_frequency,
                     wavelet_frequency_resolution,
                     title=None, savefig=None, dpi=100):
    Nt, Nf = get_shape_of_wavelet_transform(duration=duration,
                                            sampling_frequency=sampling_frequency,
                                            wavelet_frequency_resolution=wavelet_frequency_resolution)
    sampling_times = np.arange(Nt) * duration / Nt
    sampling_frequencies = np.arange(Nf) * wavelet_frequency_resolution
    cmap = plt.get_cmap("viridis")
    levels = MaxNLocator(nbins=15).tick_values(np.min(spectrogram),np.max(spectrogram))
    norm = BoundaryNorm(levels,ncolors=cmap.N,clip=True)
    fig, ax = plt.subplot()
    img = librosa.display.specshow(spectrogram.T,y_axis="log",x_axis="s",cmap=cmap,norm=norm,x_coords=sampling_times,y_coords=sampling_frequencies,snap=True, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Frequency (Hz)")
    cbar = fig.colorbar(img,format="%.0e", ax=ax)
    cbar.set_label("Normalized Energy")
    cbar.ax.yaxis.set_label_position("left")
    plt.xlabel("Time (s)")
    if savefig is not None:
        plt.savefig(savefig,dpi=dpi,bbox_inches="tight")
    plt.clf()
    plt.close()
