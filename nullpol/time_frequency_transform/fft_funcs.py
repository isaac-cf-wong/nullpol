from numba import njit
import numpy as np

@njit
def single_frequency_fourier_transform(data, k):
    """
    Computes the Fourier transform component for a specific frequency index `k` on a real-valued input array.

    Args:
        data (array-like): A real-valued input array representing the signal.
        k (int): The frequency index for which to compute the Fourier component.

    Returns:
        complex: The Fourier transform component at frequency index `k`, represented as a complex number.
    """
    N = len(data)
    result_real = 0.0
    result_imag = 0.0
    for n in range(N):
        angle = -2.0 * np.pi * k * n / N
        result_real += data[n] * np.cos(angle)
        result_imag += data[n] * np.sin(angle)
    return result_real + 1j * result_imag    