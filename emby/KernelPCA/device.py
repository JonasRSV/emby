from numba import cuda
from emby.config import Logging
from . import cpu_core, gpu_core


def detect(logging: int, kernel: str, **kwargs):
    try:
        if logging >= Logging.Everything:
            cuda.detect()

        if cuda.is_available():
            return gpu(logging, kernel, **kwargs)

    except cuda.cudadrv.error.CudaSupportError as e:
        if logging >= Logging.Everything:
            print(f"Unable to initialize cuda driver {e}")

    return cpu(logging, kernel, **kwargs)


def cpu(logging: int, kernel: str, **kwargs):
    if logging >= Logging.Everything:
        print(f"Device CPU")

    return cpu_core.fit, cpu_core.project, cpu_core.kernel(name=kernel, **kwargs)


def gpu(logging: int, kernel: str, **kwargs):
    if logging >= Logging.Everything:
        print(f"Device GPU")
    raise NotImplementedError()


