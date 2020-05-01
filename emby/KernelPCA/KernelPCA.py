import numpy as np
from emby.KernelPCA.device import detect, cpu, gpu
from emby.config import Logging, Device


class KernelPCA:
    """
    A Implementation of kernel PCA

    Parameters
    ----------
    Z
        dimensions of the embedding space
    kernel
        The feature space (gaussian | polynomial)
    logging
        :class:`emby.Logging` level of logging to use (default no logging)
    device
        :class:`emby.Device` device configuration for this class (default detect)
    ``**kwargs``
        additional arguments.. passed to kernel (variance for gaussian, c & d for polynomial)


    Examples
    ---------

    Fitting some 2D points

    >>> from emby import KernelPCA
    >>> import numpy as np
    >>> x = np.concatenate([
    ...    np.random.multivariate_normal(np.ones(2) * -4, np.eye(2), size=500),
    ...    np.random.multivariate_normal(np.ones(2) * 4, np.eye(2), size=500)
    ... ])
    >>> kpca = KernelPCA(Z=2)
    >>> kpca.fit_transform(x)

    """

    def __init__(self, Z: int,
                 kernel: str = "gaussian",
                 logging: int = Logging.Nothing,
                 device: int = Device.Detect,
                 **kwargs):
        self.z = Z

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

        self._fit, self._project, self._kernel = modes[device](logging, kernel=kernel, **kwargs)

        self.evecs = None
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

        >>> from emby import KernelPCA
        >>> import numpy as np
        >>> x = np.concatenate([
        ...    np.random.multivariate_normal(np.ones(2) * -4, np.eye(2), size=500),
        ...    np.random.multivariate_normal(np.ones(2) * 4, np.eye(2), size=500)
        ... ])
        >>> kpca = KernelPCA(Z=2)
        >>> kpca.fit(x)
        """

        self.X, self.evecs = self._fit(x, z=self.z, kernel=self._kernel)

        return self

    def fit_transform(self, x: np.ndarray):
        """
        fit_transform(self, x: np.ndarray)

        fit and then transform a tensor x onto the bases of the :class:`SOM`

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
        >>> from emby import KernelPCA
        >>> import numpy as np
        >>> x = np.concatenate([
        ...    np.random.multivariate_normal(np.ones(2) * -4, np.eye(2), size=500),
        ...    np.random.multivariate_normal(np.ones(2) * 4, np.eye(2), size=500)
        ... ])
        >>> kpca = KernelPCA(Z=2)
        >>> base_space = kpca.fit_transform(x)
        >>> plt.plot(base_space[:, 0], base_space[:, 1])\
        ... + np.random.normal(0, 1e-1, size=x.shape) # for some nice looking noise

        """

        self.fit(x)
        return self.transform(x)

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
        >>> from emby import KernelPCA
        >>> import numpy as np
        >>> x = np.concatenate([
        ...    np.random.multivariate_normal(np.ones(2) * -4, np.eye(2), size=500),
        ...    np.random.multivariate_normal(np.ones(2) * 4, np.eye(2), size=500)
        ... ])
        >>> kcpa = KernelPCA(Z=2)
        >>> kcpa.fit(x)
        >>> base_space = kcpa.transform(x)
        >>> plt.plot(base_space[:, 0], base_space[:, 1])
        ... + np.random.normal(0, 1e-1, size=x.shape) # for some nice looking noise

        """
        return self._project(x, self.X, eigen_vectors=self.evecs, kernel=self._kernel)


