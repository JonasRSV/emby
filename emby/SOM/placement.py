import itertools
import numpy as np


def _place_uniform(x: np.ndarray, bases: int, z: int, uniform_base_position="random", **kwargs) -> (
np.ndarray, np.ndarray):
    if uniform_base_position == "random":
        base_indexes = np.arange(len(x))
        x_bases = x[np.random.choice(base_indexes, size=bases)].copy()
    elif uniform_base_position == "origo":
        x_bases = np.zeros((bases, x.shape[1]))
    else:
        raise NotImplementedError(f"{uniform_base_position} is not implemented (random | origo) is")

    y_bases = np.zeros((bases, z))

    dim_size = int(np.float_power(bases, 1 / z) + 0.99)
    combinations = itertools.product(np.arange(dim_size), repeat=z)
    for i in range(bases):
        base = next(combinations)
        y_bases[i] = np.array(base)

    return x_bases, y_bases / (dim_size - 1)
