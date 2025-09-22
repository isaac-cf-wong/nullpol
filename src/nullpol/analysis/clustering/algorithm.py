from __future__ import annotations

import numpy as np


def _get_neighbours(i, j, mask):
    """Get valid 8-connected neighboring coordinates for a given pixel in a 2D mask.

    Returns all neighboring coordinates within the mask boundaries using 8-connectivity
    (including diagonal neighbors). The center pixel (i, j) is excluded from the result.

    Args:
        i (int): Row index of the center pixel.
        j (int): Column index of the center pixel.
        mask (numpy.ndarray): 2D array representing the mask boundaries.

    Returns:
        list[tuple[int, int]]: List of valid neighboring coordinate tuples (row, col).
            Maximum of 8 neighbors for interior pixels, fewer for edge/corner pixels.
    """
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
    """Perform depth-first search to identify a connected cluster in a binary mask.

    Uses an iterative depth-first search algorithm to find all connected pixels
    starting from a seed pixel (i, j). Pixels are considered connected if they
    are adjacent (8-connectivity) and both have True values in the mask.

    Args:
        i (int): Starting row index for the search.
        j (int): Starting column index for the search.
        mask (numpy.ndarray): 2D binary mask where True indicates valid pixels.
        visited (numpy.ndarray): 2D boolean array tracking visited pixels, modified in-place.

    Returns:
        list[tuple[int, int]]: List of coordinate tuples (row, col) forming a connected cluster.
    """
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
    """Find the largest connected cluster in a time-frequency filter and apply padding.

    This function identifies connected components in a time-frequency filter using
    depth-first search, selects the largest cluster, and applies symmetric padding in
    both time and frequency dimensions.

    The clustering algorithm:
    1. Finds all connected components using 8-connectivity
    2. Selects the largest cluster by number of pixels
    3. Applies symmetric padding around the largest cluster in both time and frequency directions
    4. Returns a mask covering the padded region

    Args:
        filter (numpy.ndarray): 2D mask of shape (n_time, n_freq) where True
            indicates pixels of interest (e.g., excess power regions).
        dt (float): Time resolution in seconds. Used to convert padding_time to pixel units.
        df (float): Frequency resolution in Hz. Used to convert padding_freq to pixel units.
        padding_time (float, optional): Symmetric time padding in seconds applied to both
            sides of the cluster. Defaults to 0.1.
        padding_freq (float, optional): Symmetric frequency padding in Hz applied to both
            sides of the cluster. Defaults to 10.
        **kwargs: Additional keyword arguments (unused, for compatibility).

    Returns:
        numpy.ndarray: 2D mask of shape (n_time, n_freq) with dtype uint8.
            Contains 1 for pixels within the padded largest cluster, 0 elsewhere.

    Note:
        - Uses 8-connectivity for cluster identification (includes diagonal neighbors)
        - Padding is applied symmetrically and clipped to array boundaries
        - If multiple clusters have the same maximum size, the first one found is selected
        - The function modifies a copy of the input and doesn't alter the original filter
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
