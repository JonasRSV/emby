import numpy as np
from emby.Preedy.device import detect, cpu, gpu
from emby.config import Logging, Device


class Preedy:
    """
    A Implementation of Preedy

    Parameters
    ----------
    Z
        dimensions of the embedding space
    alpha
        parameter determining "closeness" of neighbours
    beta
        Parameter determining distance of non-neighbours
    n
        Number of neighbours
    logging
        :class:`emby.Logging` level of logging to use (default no logging)
    device
        :class:`emby.Device` device configuration for this class (default detect)
    ``**kwargs``
        additional arguments


    Examples
    ---------

    Fitting some 2D points

    >>> from emby import Preedy
    >>> import numpy as np
    >>> x = np.concatenate([
    ...    np.random.multivariate_normal(np.ones(2) * -4, np.eye(2), size=500),
    ...    np.random.multivariate_normal(np.ones(2) * 4, np.eye(2), size=500)
    ... ])
    >>> preedy = Preedy(Z=2)
    >>> preedy.fit_transform(x)

    """

    def __init__(self, Z: int,
                 alpha: np.float64 = 2.0,
                 beta: np.float64 = 35.0,
                 n: int = 20,
                 logging: int = Logging.Nothing,
                 device: int = Device.Detect,
                 **kwargs):
        self.z = Z
        self.alpha = alpha
        self.beta = beta
        self.n = n

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

        self._fit, self._project = modes[device](logging, **kwargs)

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

        >>> from emby import Preedy
        >>> import numpy as np
        >>> x = np.concatenate([
        ...    np.random.multivariate_normal(np.ones(2) * -4, np.eye(2), size=500),
        ...    np.random.multivariate_normal(np.ones(2) * 4, np.eye(2), size=500)
        ... ])
        >>> preedy = Preedy(Z=2)
        >>> preedy.fit(x)
        """

        self.X = self._fit(x, z=self.z, alpha=self.alpha, beta=self.beta, n=self.n, verbose=self.fit_verbose)

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
        >>> from emby import Preedy
        >>> import numpy as np
        >>> x = np.concatenate([
        ...    np.random.multivariate_normal(np.ones(2) * -4, np.eye(2), size=500),
        ...    np.random.multivariate_normal(np.ones(2) * 4, np.eye(2), size=500)
        ... ])
        >>> preedy = Preedy(Z=2)
        >>> base_space = preedy.fit_transform(x)
        >>> plt.plot(base_space[:, 0], base_space[:, 1])

        """

        self.fit(x)
        return self.X

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
        >>> from emby import Preedy
        >>> import numpy as np
        >>> x = np.concatenate([
        ...    np.random.multivariate_normal(np.ones(2) * -4, np.eye(2), size=500),
        ...    np.random.multivariate_normal(np.ones(2) * 4, np.eye(2), size=500)
        ... ])
        >>> preedy = Preedy(Z=2)
        >>> preedy.fit(x)
        >>> base_space = preedy.transform(x)
        >>> plt.plot(base_space[:, 0], base_space[:, 1])

        """
        return self._project(x, self.X)


