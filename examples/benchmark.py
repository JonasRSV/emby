import emby
from emby.SOM.cpu_core import _euclidean_argmin as cpu_argmin
from emby.SOM.gpu_core import _euclidean_argmin_gpu as gpu_argmin
from emby.SOM.gpu_core import _fit_winners_pull_gpu as pull_gpu
import time
from numba import cuda
import numpy as np


def benchmark_argmin():
    SIZE_LG = 1000000
    SIZE_SM = 100
    DIM = 100

    a = np.random.rand(SIZE_LG, DIM).astype(np.float32)
    b = np.random.rand(SIZE_SM, DIM).astype(np.float32)
    c = np.zeros(SIZE_LG, dtype=np.int32)



    a_gpu = cuda.to_device(a)
    b_gpu = cuda.to_device(b)
    c_gpu = cuda.to_device(c)

    thread_in_batch, batches = 128, 64
    #thread_in_batch, batches = 32, 64

    cpu_argmin(a, b)

    gpu_argmin[batches, thread_in_batch](a_gpu, b_gpu, c_gpu)
    cuda.synchronize()

    #time.sleep(4)


    timestamp = time.time()
    cpu_argmin(a, b)
    print("CPU time",  time.time() - timestamp)

    timestamp = time.time()
    gpu_argmin[batches, thread_in_batch](a_gpu, b_gpu, c_gpu)
    cuda.synchronize()
    print("GPU time", time.time() - timestamp)

def benchmark_winners_pull():
    SIZE_LG = 100000
    SIZE_SM = 100
    DIM = 100

    a = np.random.rand(SIZE_LG, DIM).astype(np.float32)
    b = np.random.rand(SIZE_SM, DIM).astype(np.float32)
    c = np.zeros(SIZE_LG, dtype=np.int32)


    a_gpu = cuda.to_device(a)
    b_gpu = cuda.to_device(b)
    c_gpu = cuda.to_device(c)

    learing_rate = cuda.to_device(0.1)
    y_neighbourhood = cuda.to_device(np.random.rand(SIZE_SM, SIZE_SM))

    thread_in_batch, batches = 128, 64

    gpu_argmin[batches, thread_in_batch](a_gpu, b_gpu, c_gpu)
    cuda.synchronize()
    pull_gpu[batches, thread_in_batch](c_gpu, a_gpu, b_gpu, y_neighbourhood, learing_rate)
    cuda.synchronize()

    timestamp = time.time()
    pull_gpu[batches, thread_in_batch](c_gpu, a_gpu, b_gpu, y_neighbourhood, learing_rate)
    cuda.synchronize()
    print("GPU time", time.time() - timestamp)


#benchmark_argmin()
benchmark_winners_pull()
