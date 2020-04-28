import numba
import numpy as np
import itertools
from typing import Tuple


@numba.jit(nopython=True, fastmath=True, forceobj=False)
def _fast_euclidean(x: np.ndarray, x_bases: np.ndarray):  # 10x faster than numpy variant
    x_size, bases_size = len(x), len(x_bases)
    dim = x.shape[1]

    z = np.zeros((x_size, bases_size))

    for i in range(x_size):
        for j in range(bases_size):
            for k in range(dim):
                z[i, j] += (x[i, k] - x_bases[j, k]) * (x[i, k] - x_bases[j, k])

    return z


@numba.jit(nopython=True, fastmath=True, forceobj=False)
def _fast_euclidean_argmin(x: np.ndarray, x_bases: np.ndarray):
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
            verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
    bases = len(x_bases)

    for e in range(epochs):
        if verbose:
            x_freeze = x_bases.copy()

        winners = _fast_euclidean_argmin(x, x_bases)

        y_neighbourhood = np.exp(-np.sqrt(_fast_euclidean(y_bases, y_bases)) / y_variance)

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


def _place_uniform(x: np.ndarray, bases: int, z: int) -> (np.ndarray, np.ndarray):
    base_indexes = np.arange(len(x))
    x_bases = x[np.random.choice(base_indexes, size=bases)].copy()
    y_bases = np.zeros((bases, z))

    dim_size = int(np.float_power(bases, 1 / z) + 0.99)
    combinations = itertools.product(np.arange(dim_size), repeat=z)
    for i in range(bases):
        base = next(combinations)
        y_bases[i] = np.array(base)

    return x_bases, y_bases / (dim_size - 1)


def _place_spheres(x: np.ndarray, bases: int, z: int, spheres: int = 2, variance: int = 0.001):
    base_indexes = np.arange(len(x))
    x_bases = x[np.random.choice(base_indexes, size=bases)].copy()
    y_bases = np.zeros((bases, z))

    bases_per_sphere = bases // spheres

    i = 0
    for sph in range(spheres - 1):
        mean = np.random.rand(z)
        cov = np.eye(z) * variance
        y_bases[i: i + bases_per_sphere] = np.random.multivariate_normal(mean, cov, size=bases_per_sphere)
        i += bases_per_sphere

    mean = np.random.rand(z)
    cov = np.eye(z) * variance
    y_bases[i: bases] = np.random.multivariate_normal(mean, cov, size=bases - i)

    return x_bases, y_bases


class SOM:

    def __init__(self, z: int, bases: int,
                 learning_rate: float = 0.1,
                 epochs: int = 50,
                 y_variance=None,
                 verbose=False,
                 mode: str = "uniform",
                 **mode_kwargs):
        self.z = z
        self.bases = bases
        self.learning_rate = learning_rate
        self.epochs = epochs

        if y_variance is None: y_variance = 0.1

        self.y_variance = y_variance
        self.verbose = verbose

        modes = {
            "uniform": _place_uniform,
            "sphere": _place_spheres
        }

        self.mode = modes[mode]
        self.mode_kwargs = mode_kwargs

        self.x_bases = None
        self.y_bases = None

    def fit(self, x: np.ndarray):
        self.x_bases, self.y_bases = self.mode(x,
                                               bases=self.bases,
                                               z=self.z,
                                               **self.mode_kwargs)

        self.x_bases = _pp_fit(x,
                               x_bases=self.x_bases,
                               y_bases=self.y_bases,
                               learning_rate=self.learning_rate,
                               y_variance=self.y_variance,
                               epochs=self.epochs,
                               verbose=self.verbose)

        return self

    def fit_transform(self, x: np.ndarray):
        self.fit(x)
        return self.transform(x)

    def transform(self, x: np.ndarray):
        return self.y_bases[_fast_euclidean_argmin(x, self.x_bases)]

    def base_similarities(self, variance: float = 4.0):
        distances = np.sqrt(_fast_euclidean(self.x_bases, self.x_bases))
        return np.exp(-distances / variance)
