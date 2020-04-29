from numba import cuda
from emby.config import Logging
from . import cpu_core
import platform


def detect(logging: int):
    try:
        device_info = cuda.detect()

        print("Device info", device_info)

        if cuda.is_available():
            return gpu(logging)

    except cuda.cudadrv.error.CudaSupportError as e:
        if logging >= Logging.Everything:
            print(f"Unable to initialize cuda driver {e}")

    return cpu(logging)


def cpu(logging: int):
    if logging >= Logging.Everything:
        print(f"Device CPU according to python platform: {platform.processor()}")

    return cpu_core._pp_fit, cpu_core._euclidean_argmin, cpu_core._similarity_mode


def gpu(logging: int):
    raise NotImplementedError("GPU is not implemented yet")
