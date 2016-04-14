import numpy as np

from neupy.core.properties import NumberProperty
from neupy.core.base import BaseSkeleton
from neupy.utils import as_array2d, NotTrainedException


__all__ = ('ZCA',)


class ZCA(BaseSkeleton):
    """ ZCA (zero-phase component analysis) whitening.

    Parameters
    ----------
    regularization : float
        Regularization parameter. Defaults to ``1e-5``.

    Attributes
    ----------
    mean : 1D array
        Mean for each feature.
    components : array-like
        ZCA components.

    Methods
    -------
    train(data)
        Train ZCA.
    transform(data)
        Transform input data.
    """
    regularization = NumberProperty(default=1e-5, minval=0)

    def __init__(self, regularization=1e-5, **options):
        self.regularization = regularization
        self.mean = None
        self.components = None
        super(ZCA, self).__init__(**options)

    def fit(self, X, *args, **kwargs):
        """ This method is an alias to `train` method.
        This method is important for the scikit-learn
        compatibility.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        ZCA class instance
        """
        self.train(X, *args, **kwargs)
        return self

    def train(self, data):
        """ Train ZCA.

        Parameters
        ----------
        data : array-like
        """
        data = as_array2d(data)
        self.mean = data.mean(axis=0)
        data = data - self.mean

        n_features = data.shape[1]
        sigma = np.dot(data.T, data) / n_features
        U, S, V = np.linalg.svd(sigma)

        self.components = (U / np.sqrt(S + self.regularization)).dot(U.T)

    def transform(self, data):
        """ Apply ZCA transformation on data.

        Parameters
        ----------
        data : array-like

        Returns
        -------
        array-like
        """
        if self.mean is None or self.components is None:
            raise NotTrainedException("Train ZCA before use it.")

        data_shape = data.shape
        data = as_array2d(data)
        data_transformed = data - self.mean
        data_transformed = np.dot(data_transformed, self.components.T)
        return data_transformed.reshape(data_shape)
