import itertools
import numpy as np


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
