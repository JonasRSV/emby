import numpy as np
from emby.MM.device import detect, cpu, gpu
from emby.config import Logging, Device


class MM:
    """
    A Implementation of metric minimization

    Parameters
    ----------
    Z
        dimensions of the embedding space
    metric:
        metric to minimize: euclidean | cosine
    x_variance:
        variance in x_space
    z_variance:
        variance in z_space
    logging
        :class:`emby.Logging` level of logging to use (default no logging)
    device
        :class:`emby.Device` device configuration for this class (default detect)
    ``**kwargs``
        additional arguments.. passed to kernel (variance for gaussian, c & d for polynomial)


    Examples
    ---------

    Fitting some 2D points

    >>> from emby import MM
    >>> import numpy as np
    >>> x = np.concatenate([
    ...    np.random.multivariate_normal(np.ones(2) * -4, np.eye(2), size=500),
    ...    np.random.multivariate_normal(np.ones(2) * 4, np.eye(2), size=500)
    ... ])
    >>> mm = MM(Z=2)
    >>> mm.fit_transform(x)

    """

    def __init__(self, Z: int,
                 metric:str = "euclidean",
                 x_variance: float = 1.0,
                 z_variance: float = 1.0,
                 epochs: int = 100,
                 logging: int = Logging.Nothing,
                 device: int = Device.Detect,
                 **kwargs):
        self.z = Z
        self.x_variance = x_variance
        self.z_variance = z_variance
        self.epochs = epochs

        self.logging = logging

        self.fit_verbose = False
        if logging > Logging.Nothing:
            self.fit_verbose = True

        self.kwargs = kwargs

        modes = {
            Device.Detect: detect,
            Device.CPU: cpu,
            Device.GPU: gpu
        }

        self._fit, self._closest = modes[device](logging, metric=metric, **kwargs)

        self.bases = None
        self.X = None

    def fit(self, x: np.ndarray):
        """
        fit(self, x: np.ndarray)

        Parameters
        ----------
        x
            A 2-dimensional ndarray N x M

        Returns
        -------
        ndarray
            A 2-dimensional array N x Z

        Examples
        --------

        Fit some 2D data

        >>> from emby import MM
        >>> import numpy as np
        >>> x = np.concatenate([
        ...    np.random.multivariate_normal(np.ones(2) * -4, np.eye(2), size=500),
        ...    np.random.multivariate_normal(np.ones(2) * 4, np.eye(2), size=500)
        ... ])
        >>> mm = MM(Z=2)
        >>> mm.fit(x)
        """

        self.bases = self._fit(x=x,
                               z=self.z,
                               x_variance=self.x_variance,
                               z_variance=self.z_variance,
                               epochs=self.epochs,
                               verbose=self.fit_verbose)
        self.X = x

        return self


    def fit_transform(self, x: np.ndarray):
        """
        fit_transform(self, x: np.ndarray)

        Parameters
        ----------
        x
            A 2-dimensional ndarray N x M

        Returns
        -------
        ndarray
            A 2-dimensional array N x Z

        Examples
        --------

        Fit, transform and plot some 2D data

        >>> import matplotlib.pyplot as plt
        >>> from emby import MM
        >>> import numpy as np
        >>> x = np.concatenate([
        ...    np.random.multivariate_normal(np.ones(2) * -4, np.eye(2), size=500),
        ...    np.random.multivariate_normal(np.ones(2) * 4, np.eye(2), size=500)
        ... ])
        >>> mm = MM(Z=2)
        >>> base_space = mm.fit_transform(x)
        >>> plt.plot(base_space[:, 0], base_space[:, 1])

        """

        self.fit(x)
        return self.bases

    def transform(self, x: np.ndarray):
        """
        transform(self, x: np.ndarray)

        Parameters
        ----------
        x
            A 2-dimensional ndarray N x M

        Returns
        -------
        ndarray
            A 2-dimensional array N x Z

        Examples
        --------

        transform and plot some 2D data

        >>> import matplotlib.pyplot as plt
        >>> from emby import MM
        >>> import numpy as np
        >>> x = np.concatenate([
        ...    np.random.multivariate_normal(np.ones(2) * -4, np.eye(2), size=500),
        ...    np.random.multivariate_normal(np.ones(2) * 4, np.eye(2), size=500)
        ... ])
        >>> mm = MM(Z=2)
        >>> mm.fit(x)
        >>> base_space = mm.transform(x)
        >>> plt.plot(base_space[:, 0], base_space[:, 1])

        """

        return self.bases[self._closest(x, self.X)]
