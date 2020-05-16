from numba import cuda
from emby.config import Logging
from . import cpu_core, gpu_core


def detect(logging: int, **kwargs):
    try:
        if logging >= Logging.Everything:
            cuda.detect()

        if cuda.is_available():
            return gpu(logging, **kwargs)

    except cuda.cudadrv.error.CudaSupportError as e:
        if logging >= Logging.Everything:
            print(f"Unable to initialize cuda driver {e}")

    return cpu(logging, **kwargs)


def cpu(logging: int, **kwargs):
    if logging >= Logging.Everything:
        print(f"Device CPU")

    return cpu_core.fit, cpu_core.project


def gpu(logging: int, **kwargs):
    if logging >= Logging.Everything:
        print(f"Device GPU")
    raise NotImplementedError()


