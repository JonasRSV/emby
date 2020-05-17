import math
from numba import cuda, float32
import numpy as np
import time


@cuda.jit
def _euclidean_gpu(arr_1: cuda.cudadrv.devicearray,
                   arr_2: cuda.cudadrv.devicearray,
                   res: cuda.cudadrv.devicearray):
    arr_1_size, dim = arr_1.shape
    arr_2_size = len(arr_2)

    arr_1_pos = cuda.grid(1)
    arr_1_grid_size = cuda.gridsize(1)

    for i in range(arr_1_pos, arr_1_size, arr_1_grid_size):
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

    threads_per_block = 128
    blocks = 64

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
    grid_size = cuda.gridsize(1)

    
    local = cuda.local.array(512000 // 32, dtype=float32) # max size 512kb
    # using local memory gives a 2x speed-up
    # But it limits the dimension of the bases to 60 000
    # I don't believe this will be a problem though
    for i in range(arr_1_pos, arr_1_size, grid_size):
        min_distance, min_j = 1000000.0, 0

        for k in range(dim): local[k] = arr_1[i, k] # pre-load

        
        for j in range(arr_2_size):
            zj = 0.0
            for k in range(dim):
                a1, a2 = local[k], arr_2[j, k]
                zj += (a1 - a2) * (a1 - a2)

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

    res = cuda.device_array(arr_1_size, dtype=np.int32)

    threads_per_block = 128
    blocks = 64
    _euclidean_argmin_gpu[blocks, threads_per_block](arr_1, arr_2, res)

    cuda.synchronize()

    return res.copy_to_host()


@cuda.jit
def _fit_winners_pull_gpu(winners: cuda.cudadrv.devicearray,
                          x: cuda.cudadrv.devicearray,
                          x_bases: cuda.cudadrv.devicearray,
                          y_neighbourhood: cuda.cudadrv.devicearray,
                          learning_rate: cuda.cudadrv.devicearray):
    winners_size = len(winners)
    bases_size, dim = x_bases.shape

    winner_pos = cuda.grid(1)
    grid_size = cuda.gridsize(1)

    lr = learning_rate[0] # 20 % speedup by accesing that here

    local = cuda.local.array(512000 // 64, dtype=float32) # 15% speed-up
    for i in range(winner_pos, winners_size, grid_size):
        winner = winners[i]

        for k in range(dim): local[k] = x[i, k] # pre-load

        for j in range(bases_size):
            neighbourhood = y_neighbourhood[j, winner]

            for k in range(dim):
                xb = x_bases[j, k]
                x_bases[j, k] = xb + neighbourhood * lr * (local[k] - xb)


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

    x = cuda.to_device(x.astype(np.float32))
    x_bases = cuda.to_device(x_bases.astype(np.float32))
    learning_rate = cuda.to_device(np.array(learning_rate, dtype=np.float32))

    x_size = len(x)

    threads_per_block = 128
    blocks = 64

    winners = cuda.device_array(len(x), dtype=np.int64)


    previous = np.zeros(x_size)
    avg_movement = 0.0
    for e in range(1, epochs):
        timestamp = time.time()

        _euclidean_argmin_gpu[blocks, threads_per_block](x, x_bases, winners)

        # 1. x_bases winners is pulled towards the x
        # 2. x_bases pulls x_neighbourhood towards x as well

        _fit_winners_pull_gpu[blocks, threads_per_block](winners, x, x_bases, y_neighbourhood, learning_rate)

        cuda.synchronize()
        if verbose:
            cpu_winners = winners.copy_to_host()
            avg_movement += (cpu_winners != previous).mean()
            previous = cpu_winners
            print("epoch", e, " / ", epochs, "-- movement: %.2f " % (avg_movement / e), " -- time: %.3f " % (time.time() - timestamp), end="                  \r")

    return x_bases.copy_to_host()
