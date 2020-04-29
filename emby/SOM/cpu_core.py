import numba
import numpy as np


@numba.jit(nopython=True, fastmath=True, forceobj=False)
def _euclidean(x: np.ndarray, x_bases: np.ndarray):  # 10x faster than numpy variant
    x_size, bases_size = len(x), len(x_bases)
    dim = x.shape[1]

    z = np.zeros((x_size, bases_size))

    for i in range(x_size):
        for j in range(bases_size):
            for k in range(dim):
                z[i, j] += (x[i, k] - x_bases[j, k]) * (x[i, k] - x_bases[j, k])

    return z


@numba.jit(nopython=True, fastmath=True, forceobj=False)
def _euclidean_argmin(x: np.ndarray, x_bases: np.ndarray):
    x_size, bases_size = len(x), len(x_bases)
    dim = x.shape[1]

    z = np.zeros(x_size, dtype=np.int64)

    for i in range(x_size):
        min_distance, min_j = 1e6, 0
        for j in range(bases_size):
            zj = 0.0
            for k in range(dim):
                zj += (x[i, k] - x_bases[j, k]) * (x[i, k] - x_bases[j, k])

            if zj < min_distance:
                min_distance, min_j = zj, j

        z[i] = min_j

    return z


@numba.jit(nopython=True, fastmath=True, forceobj=False)
def _pp_fit(x: np.ndarray,
            x_bases: np.ndarray,
            y_bases: np.ndarray,
            learning_rate: float,
            y_variance: float,
            epochs: int,
            verbose: bool) -> np.ndarray:
    bases = len(x_bases)

    for e in range(epochs):
        if verbose:
            x_freeze = x_bases.copy()

        winners = _euclidean_argmin(x, x_bases)

        y_neighbourhood = np.exp(-np.sqrt(_euclidean(y_bases, y_bases)) / y_variance)

        # 1. x_bases winners is pulled towards the x
        # 2. x_bases pulls x_neighbourhood towards x as well

        for i in range(winners.size):
            winner = winners[i]

            for j in range(bases):
                neighbourhood = y_neighbourhood[j][winner]
                x_bases[j][:] = x_bases[j] + neighbourhood * learning_rate * (x[i] - x_bases[j])

        if verbose:
            print("epoch", e, " / ", epochs, " -- ", np.abs(x_bases - x_freeze).sum())

    return x_bases


@numba.jit(nopython=True, fastmath=True, forceobj=False)
def _similarity_mode(x_bases: np.ndarray):
    distances = np.sqrt(_euclidean(x_bases, x_bases))
    variances = np.exp(np.linspace(-6, 10, 10000))

    bases = len(x_bases)

    similarity_mode, mode = None, -1.0
    for var in variances:
        mat = np.exp(-distances / var)

        mean_similarity = 0.0
        for i in range(bases):
            for j in range(bases):
                if i != j:
                    mean_similarity += mat[i, j]

        mean_similarity = mean_similarity / (bases * bases - bases)

        variance_similarity = 0.0
        for i in range(bases):
            for j in range(bases):
                if i != j:
                    variance_similarity += np.square(mat[i, j] - mean_similarity)

        if variance_similarity > mode:
            similarity_mode = mat
            mode = variance_similarity

    return similarity_mode




