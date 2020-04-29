from numba import cuda
from emby.config import Logging
from . import cpu_core, gpu_core


def detect(logging: int):
    try:
        cuda.detect()
        if cuda.is_available():
            return gpu(logging)

    except cuda.cudadrv.error.CudaSupportError as e:
        if logging >= Logging.Everything:
            print(f"Unable to initialize cuda driver {e}")

    return cpu(logging)


def cpu(logging: int):
    if logging >= Logging.Everything:
        print(f"Device CPU")

    return cpu_core._fit, cpu_core._euclidean_argmin, cpu_core._similarity_mode


def gpu(logging: int):
    if logging >= Logging.Everything:
        print(f"Device GPU")

    return gpu_core._fit, gpu_core._euclidean_argmin, cpu_core._similarity_mode

