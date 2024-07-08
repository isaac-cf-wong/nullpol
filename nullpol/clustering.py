import numpy as np

# Get the neighbours of a cell
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

def clustering(time_freq_transformed, threshold=90):
    """
    Clusters the time-frequency transformed data and returns a mask with the largest cluster.

    Parameters
    ----------
    time_freq_transformed : numpy.ndarray
        Time-frequency transformed data in shape (n_time, n_freq).
    threshold : int, optional
        The threshold for the clustering. Default is 10.

    Returns
    -------
    numpy.ndarray
        A mask with the largest cluster in shape (n_time, n_freq).
    """
    # threshold the data
    mask = time_freq_transformed > np.percentile(time_freq_transformed, threshold)

    # find clusters
    visited = np.zeros(mask.shape, dtype=np.uint8)
    clusters = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] and not visited[i, j]:
                clusters.append(_dfs(i, j, mask, visited))

    # return a mask with the largest cluster
    largest_cluster = max(clusters, key=len)
    mask = np.zeros(mask.shape, dtype=np.uint8)
    for i, j in largest_cluster:
        mask[i, j] = 1
    return mask
