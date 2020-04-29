import math
from numba import cuda
import numpy as np


@cuda.jit
def _euclidean_gpu(arr_1: cuda.cudadrv.devicearray,
                   arr_2: cuda.cudadrv.devicearray,
                   res: cuda.cudadrv.devicearray):
    arr_1_size, dim = arr_1.shape
    arr_2_size = len(arr_2)

    arr_1_pos = cuda.grid(1)
    arr_1_block_size = cuda.gridsize(1)

    for i in range(arr_1_pos, arr_1_size, arr_1_block_size):
        for j in range(arr_2_size):
            for k in range(dim):
                res[i, j] += (arr_1[i, k] - arr_2[j, k]) * (arr_1[i, k] - arr_2[j, k])


def _euclidean(arr_1: np.ndarray, arr_2: np.ndarray):
    arr_1, arr_2 = np.ascontiguousarray(arr_1, dtype=np.float32), \
                   np.ascontiguousarray(arr_2, dtype=np.float32)

    arr_1 = cuda.to_device(arr_1)
    arr_2 = cuda.to_device(arr_2)

    arr_1_size = len(arr_1)
    arr_2_size = len(arr_2)

    res = cuda.device_array((arr_1_size, arr_2_size), dtype=np.float32)

    threads_per_block = np.minimum(1024, arr_1_size)
    blocks = np.maximum(math.ceil(arr_1_size / threads_per_block), 1)

    _euclidean_gpu[blocks, threads_per_block](arr_1, arr_2, res)

    cuda.synchronize()

    return res.copy_to_host()


@cuda.jit
def _euclidean_argmin_gpu(arr_1: cuda.cudadrv.devicearray,
                          arr_2: cuda.cudadrv.devicearray,
                          res: cuda.cudadrv.devicearray):
    arr_1_size, dim = arr_1.shape
    arr_2_size = len(arr_2)

    arr_1_pos = cuda.grid(1)
    block_size = cuda.gridsize(1)

    for i in range(arr_1_pos, arr_1_size, block_size):
        min_distance, min_j = 10000, 0
        for j in range(arr_2_size):
            zj = 0.0
            for k in range(dim):
                zj += (arr_1[i, k] - arr_2[j, k]) * (arr_1[i, k] - arr_2[j, k])

            if zj < min_distance:
                min_distance, min_j = zj, j

        res[i] = min_j


def _euclidean_argmin(arr_1: np.ndarray, arr_2: np.ndarray):
    arr_1, arr_2 = np.ascontiguousarray(arr_1, dtype=np.float32), \
                   np.ascontiguousarray(arr_2, dtype=np.float32)

    arr_1 = cuda.to_device(arr_1)
    arr_2 = cuda.to_device(arr_2)

    arr_1_size = len(arr_1)
    arr_2_size = len(arr_2)

    res = cuda.device_array(arr_1_size, dtype=np.int)

    threads_per_block = np.minimum(1024, arr_1_size)
    blocks = np.maximum(math.ceil(arr_1_size / threads_per_block), 1)

    _euclidean_argmin_gpu[blocks, threads_per_block](arr_1, arr_2, res)

    cuda.synchronize()
    return res.copy_to_host()


@cuda.jit
def _fit_winners_pull_gpu(winners: cuda.cudadrv.devicearray,
                          x: cuda.cudadrv.devicearray,
                          x_bases: cuda.cudadrv.devicearray,
                          learning_rate: cuda.cudadrv.devicearray,
                          y_neighbourhood: cuda.cudadrv.devicearray):
    winners_size = len(winners)
    bases_size, dim = x_bases.shape

    winner_pos = cuda.grid(1)
    block_size = cuda.gridsize(1)

    for i in range(winner_pos, winners_size, block_size):
        winner = winners[i]

        for j in range(bases_size):
            neighbourhood = y_neighbourhood[j, winner]

            for k in range(dim):
                x_bases[j, k] = x_bases[j, k] + neighbourhood * learning_rate[0] * (x[i, k] - x_bases[j, k])


def _fit(x: np.ndarray,
         x_bases: np.ndarray,
         y_bases: np.ndarray,
         learning_rate: float,
         y_variance: float,
         epochs: int,
         verbose: bool) -> np.ndarray:
    # Running this on CPU since its a one-time calculation that is not particularly slow
    # bases tends to be << x
    y_neighbourhood = cuda.to_device(
        np.exp(
            -np.sqrt(_euclidean(y_bases, y_bases)) / y_variance
        ).astype(np.float32))

    x = cuda.to_device(x.astype(np.float32))  # single precision because its like 20x speed-up on nvidia gpus
    x_bases = cuda.to_device(x_bases.astype(np.float32))
    learning_rate = cuda.to_device(np.array(learning_rate, dtype=np.float32))

    x_size = len(x)

    threads_per_block = np.minimum(1024, x_size)
    blocks = np.maximum(math.ceil(x_size / threads_per_block), 1)

    winners = cuda.device_array(len(x), dtype=np.int)

    for e in range(epochs):
        _euclidean_argmin_gpu[blocks, threads_per_block](x, x_bases, winners)

        # 1. x_bases winners is pulled towards the x
        # 2. x_bases pulls x_neighbourhood towards x as well

        _fit_winners_pull_gpu[blocks, threads_per_block](winners, x, x_bases, learning_rate, y_neighbourhood)

        if verbose:
            print("epoch", e, " / ", epochs)

    cuda.synchronize()
    return x_bases.copy_to_host()
