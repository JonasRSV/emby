import numpy as np
from emby.SOM.placement import _place_spheres, _place_uniform
from emby.SOM.core import _pp_fit, _euclidean_argmin, _similarity_mode


class SOM:
    """
    A self-organizing map

    Parameters
    ----------
    Z
        dimensions of the embedding space
    bases
        Number of bases in the map
    learning-rate
        Learning rate used in competitive learning
    epochs
        Number of competitive learning iterations
    y_variance
        variance of neighbourhood function in embedding space
    verbose
        if true, prints progress of competitive learning
    mode
        ("uniform" | "sphere") how the bases are placed in the embedding space
    ``**mode_kwargs``
        additional arguments for different modes


    Examples
    ---------

    Fitting some 2D points

    >>> from emby import SOM
    >>> x = np.concatenate([
    ...    np.random.multivariate_normal(np.ones(2) * -4, np.eye(2), size=500),
    ...    np.random.multivariate_normal(np.ones(2) * 4, np.eye(2), size=500)
    ... ])
    >>> som = SOM(Z=2, bases=2)
    >>> som.fit(x)

    """

    def __init__(self, Z: int, bases: int,
                 learning_rate: float = 0.01,
                 epochs: int = 30,
                 y_variance: float = 0.1,
                 verbose: bool = False,
                 mode: str = "uniform",
                 **mode_kwargs):
        self.z = Z
        self.bases = bases
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.y_variance = y_variance
        self.verbose = verbose

        modes = {
            "uniform": _place_uniform,
            "sphere": _place_spheres
        }

        self.mode = modes[mode]
        self.mode_kwargs = mode_kwargs

        self.x_bases = None
        self.y_bases = None

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

        >>> from emby import SOM
        >>> x = np.concatenate([
        ...    np.random.multivariate_normal(np.ones(2) * -4, np.eye(2), size=500),
        ...    np.random.multivariate_normal(np.ones(2) * 4, np.eye(2), size=500)
        ... ])
        >>> som = SOM(Z=2, bases=2)
        >>> som.fit(x)
        """

        self.x_bases, self.y_bases = self.mode(x,
                                               bases=self.bases,
                                               z=self.z,
                                               **self.mode_kwargs)

        self.x_bases = _pp_fit(x,
                               x_bases=self.x_bases,
                               y_bases=self.y_bases,
                               learning_rate=self.learning_rate,
                               y_variance=self.y_variance,
                               epochs=self.epochs,
                               verbose=self.verbose)

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
        >>> from emby import SOM
        >>> x = np.concatenate([
        ...    np.random.multivariate_normal(np.ones(2) * -4, np.eye(2), size=500),
        ...    np.random.multivariate_normal(np.ones(2) * 4, np.eye(2), size=500)
        ... ])
        >>> som = SOM(Z=2, bases=2)
        >>> base_space = som.fit_transform(x)
        >>> plt.plot(base_space[:, 0], base_space[:, 1])\
        ... + np.random.normal(0, 1e-1, size=x.shape) # for some nice looking noise

        """

        self.fit(x)
        return self.transform(x)

    def transform(self, x: np.ndarray):
        """
        transform(self, x: np.ndarray)

        Transform a tensor x onto the bases of the :class:`SOM`

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
        >>> from emby import SOM
        >>> x = np.concatenate([
        ...    np.random.multivariate_normal(np.ones(2) * -4, np.eye(2), size=500),
        ...    np.random.multivariate_normal(np.ones(2) * 4, np.eye(2), size=500)
        ... ])
        >>> som = SOM(Z=2, bases=2)
        >>> som.fit(x)
        >>> base_space = som.transform(x)
        >>> plt.plot(base_space[:, 0], base_space[:, 1])
        ... + np.random.normal(0, 1e-1, size=x.shape) # for some nice looking noise

        """
        return self.y_bases[self.closest_base(x)]

    def closest_base(self, x: np.ndarray):
        """
        closest_base(self, x: np.ndarray)

        Find the closest bases of the vectors in the x tensor

        Parameters
        ----------
        x
            A 2-dimensional ndarray N x M

        Returns
        -------
        ndarray
            A 1-dimensional array with shape N

        Examples
        --------

        fit and find closest bases

        >>> from emby import SOM
        >>> x = np.concatenate([
        ...    np.random.multivariate_normal(np.ones(2) * -4, np.eye(2), size=5),
        ...    np.random.multivariate_normal(np.ones(2) * 4, np.eye(2), size=5)
        ... ])
        >>> som = SOM(Z=2, bases=2)
        >>> som.fit(x)
        >>> closest = som.closest_base(x)
        >>> print(closest)
        [0 0 0 0 0 1 1 1 1 1] # closest

        """

        return _euclidean_argmin(x, self.x_bases)

    def base_similarities(self):
        """
        base_similarities(self)

        returns similarities between the bases

        Returns
        -------
        ndarray
            A matrix of similarities for the bases with shape (bases x bases). The n'th entry in the similarity
            matrix is the n't base as given by the function :func:`SOM.closest_base`

        Examples
        --------

        fit and get similarities

        >>> from emby import SOM
        >>> x = np.concatenate([
        ...    np.random.multivariate_normal(np.ones(2) * -4, np.eye(2), size=5),
        ...    np.random.multivariate_normal(np.ones(2) * 4, np.eye(2), size=5)
        ... ])
        >>> som = SOM(Z=2, bases=3)
        >>> som.fit(x)
        >>> similarities = som.base_similarities(x)
        >>> print(similarities)
            [[1.         0.69387164 0.14295189]
             [0.69387164 1.         0.1022911 ]
             [0.14295189 0.1022911  1.        ]]

        2 of the bases has high covariance, which makes sense since there are only 2 underlying clusters

        """

        return _similarity_mode(self.x_bases)
