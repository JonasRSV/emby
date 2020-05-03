from numba import cuda
from emby.config import Logging
from . import cpu_core, gpu_core


def detect(logging: int, metric="euclidean", **kwargs):
    try:
        if logging >= Logging.Everything:
            cuda.detect()

        if cuda.is_available():
            return gpu(logging, metric=metric, **kwargs)

    except cuda.cudadrv.error.CudaSupportError as e:
        if logging >= Logging.Everything:
            print(f"Unable to initialize cuda driver {e}")

    return cpu(logging, metric=metric, **kwargs)


def cpu(logging: int, metric="euclidean", **kwargs):
    if logging >= Logging.Everything:
        print(f"Device CPU")

    implementation = {
        "euclidean": (cpu_core.euclidean_fit, cpu_core.euclidean_closest)
    }

    if metric in implementation:
        return implementation[metric]

    raise Exception(f"Metric {metric} not implemented -- one of {' '.join(implementation.keys())}")


def gpu(logging: int, metric="euclidean", **kwargs):
    if logging >= Logging.Everything:
        print(f"Device GPU")
    raise NotImplementedError()


