import numpy as np
import numba
import matplotlib.pyplot as plt
import time
import sys


@numba.jit(nopython=True, fastmath=True, forceobj=False)
def _distance_euclidean(base: np.ndarray, targets: np.ndarray):
    size, dim = targets.shape

    distances = np.zeros(size)
    for t in range(size):
        distance = 0.0
        for d in range(dim):
            distance += np.square(base[d] - targets[t, d])

        distances[t] = np.sqrt(distance)

    return distances


@numba.jit(nopython=True, fastmath=True, forceobj=False)
def _distance_cosine(base: np.ndarray, targets: np.ndarray):
    size, dim = targets.shape

    base_norm = np.sqrt(base @ base)
    distances = np.zeros(size)
    for t in range(size):
        norm, target_norm = 0.0, 0.0
        for d in range(dim):
            norm += base[d] * targets[t, d]
            target_norm += targets[t, d] * targets[t, d]

        target_norm = np.sqrt(target_norm)

        distances[t] = norm / (base_norm * target_norm + 0.001)

    return distances


@numba.jit(nopython=True, fastmath=True, forceobj=False)
def _gradient_euclidean(base: np.ndarray, target: np.ndarray, metric: np.ndarray):
    d_similarity_d_metric = 1 / (2 * np.sqrt(
        metric) + 0.01)
    d_metric_dz = (base - target) * -2

    return d_similarity_d_metric * d_metric_dz


@numba.jit(nopython=True, fastmath=True, forceobj=False)
def euclidean_fit(x: np.ndarray, z: int,
                  x_variance: float, z_variance: float, epochs: int,
                  verbose: bool):
    samples, dim = x.shape
    bases = np.random.normal(0, z_variance, (samples, z))

    indexes = np.arange(samples)
    gradients = np.zeros((indexes.size, z))

    """# Debugging
    
    fig, ax = plt.subplots(figsize=(10, 10))
    scat = ax.scatter(bases[:, 0], bases[:, 1])

    ymin, ymax = -5, 5
    xmin, xmax = -5, 5

    ax.set_ylim([xmin, xmax])
    ax.set_xlim([ymin, ymax])
    """
    for e in range(epochs):
        """# Very nice for debugging
        if e % 5 == 0:
            if e % 10 == 0:
                xmin, xmax = bases[:, 0].min(), bases[:, 0].max()
                ymin, ymax = bases[:, 1].min(), bases[:, 1].max()
                ax.set_xlim([xmin, xmax])
                ax.set_ylim([ymin, ymax])

            scat.set_offsets(bases)
            plt.pause(0.001)
            time.sleep(0.1)
        """

        base = np.random.choice(indexes)

        base_x = x[base]
        targets_x = x

        base_z = bases[base]
        targets_z = bases

        similarities_x = np.exp(-_distance_euclidean(base_x, targets_x) / x_variance)

        metric_z = _distance_euclidean(base_z, targets_z)  # needed for gradient
        similarities_z = np.exp(-metric_z / z_variance)

        for i in range(indexes.size):
            d_err_similarity = -2 * (similarities_x[i] - similarities_z[i])
            d_similarity_dy = _gradient_euclidean(base_z, targets_z[i], metric_z[i]) * -(1 / z_variance)

            gradients[i] = d_err_similarity * d_similarity_dy

        bases = bases - np.maximum(1 - (e / epochs), 0.05) * gradients

        if verbose and e % (epochs // 10) == 0:
            print("epoch ", e, " / ", epochs, " x ", similarities_x.mean(), " z ", similarities_z.mean())

    return bases


@numba.jit(nopython=True, fastmath=True, forceobj=False)
def euclidean_closest(x: np.ndarray, X: np.ndarray):
    x_size, bases_size = len(x), len(X)
    dim = x.shape[1]

    z = np.zeros(x_size, dtype=np.int64)

    for i in range(x_size):
        min_distance, min_j = 1e6, 0
        for j in range(bases_size):
            zj = 0.0
            for k in range(dim):
                zj += (x[i, k] - X[j, k]) * (x[i, k] - X[j, k])

            if zj < min_distance:
                min_distance, min_j = zj, j

        z[i] = min_j

    return z
