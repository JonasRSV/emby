from typing import Callable
import numpy as np
import numba


def gaussian(variance: float = 1.0, **kwargs):
    @numba.jit(nopython=True, fastmath=True, forceobj=False)
    def kernel(x, y):
        return np.exp(-np.sqrt(np.square(x - y).sum()) / variance)

    return kernel


def polynomial(c: float = 1.0, d: int = 3.0, **kwargs):
    @numba.jit(nopython=True, fastmath=True, forceobj=False)
    def kernel(x, y):
        return (x @ y + c) ** d

    return kernel


def kernel(name: str, **kwargs):
    kernels = {
        "gaussian": gaussian,
        "polynomial": polynomial
    }

    if name in kernels:
        return kernels[name](**kwargs)

    raise ValueError(f"Kernel {name} not implemented -- pick one of {' '.join(kernels.keys())}")


@numba.jit(nopython=True, fastmath=True, forceobj=False)
def project(x: np.ndarray,
            X: np.ndarray,
            eigen_vectors: np.ndarray,
            kernel: Callable[[np.ndarray, np.ndarray], float]):
    x_sz = len(x)
    N = len(X)
    z = len(eigen_vectors)

    projections = np.zeros((x_sz, z))
    for j in range(x_sz):
        for i in range(z):
            for n in range(N):
                projections[j, i] += eigen_vectors[i, n] * kernel(x[j], X[n])

    return projections


@numba.jit(nopython=True, fastmath=True, forceobj=False)
def fit(x: np.ndarray, z: int, kernel: Callable[[np.ndarray, np.ndarray], float]):
    points, dim = x.shape

    K = np.zeros((points, points))
    for i in range(points):
        for j in range(points):
            K[i, j] = kernel(x[i], x[j])

    _, vecs = np.linalg.eigh(K)

    return x, vecs[:, -z:].T
