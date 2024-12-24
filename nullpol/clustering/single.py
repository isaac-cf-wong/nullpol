import numpy as np


def _get_neighbours(i, j, mask):
    neighbours = []
    for x in range(-1, 2):
        for y in range(-1, 2):
            if x == 0 and y == 0:
                continue
            if 0 <= i + x < mask.shape[0] and 0 <= j + y < mask.shape[1]:
                neighbours.append((i + x, j + y))
    return neighbours

# Depth-first search
def _dfs(i, j, mask, visited):
    stack = [(i, j)]
    cluster = []
    while stack:
        i, j = stack.pop()
        if visited[i, j]:
            continue
        visited[i, j] = 1
        cluster.append((i, j))
        for neighbour in _get_neighbours(i, j, mask):
            if mask[neighbour[0], neighbour[1]]:
                stack.append(neighbour)
    return cluster

def clustering(filter, dt, df, padding_time=0.1, padding_freq=10, **kwargs):
    """
    Find the largest cluster in the filter.

    Parameters
    ----------
    filter : numpy.ndarray
        A binary mask in shape (n_time, n_freq).
    dt : float
        The time resolution in seconds.
    df : float
        The frequency resolution in Hz.
    padding_time : float, optional
        The padding in time direction in seconds. Default is 0.1.
    padding_freq : float, optional
        The padding in frequency direction in Hz. Default is 10.

    Returns
    -------
    numpy.ndarray
        A mask with the largest cluster in shape (n_time, n_freq).
    """

    # find clusters
    visited = np.zeros(filter.shape, dtype=np.uint8)
    clusters = []
    for i in range(filter.shape[0]):
        for j in range(filter.shape[1]):
            if filter[i, j] and not visited[i, j]:
                clusters.append(_dfs(i, j, filter, visited))

    # find the largest cluster
    largest_cluster = max(clusters, key=len)
    mask = np.zeros(filter.shape, dtype=np.uint8)
    for i, j in largest_cluster:
        mask[i, j] = 1

    # add padding
    padding_time = int(np.ceil(padding_time / dt))
    padding_freq = int(np.ceil(padding_freq / df))
    for i, j in largest_cluster:
        for x in range(-padding_time, padding_time + 1):
            for y in range(-padding_freq, padding_freq + 1):
                if 0 <= i + x < mask.shape[0] and 0 <= j + y < mask.shape[1]:
                    mask[i + x, j + y] = 1
    
    return mask
