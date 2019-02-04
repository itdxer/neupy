from numpy import dot

from neupy.utils import format_data
from neupy.exceptions import NotTrained
from neupy.core.properties import BoundedProperty
from neupy.algorithms.base import BaseSkeleton
from .utils import pdf_between_data


__all__ = ('GRNN',)


class GRNN(BaseSkeleton):
    """
    Generalized Regression Neural Network (GRNN). Network applies
    only to the regression problems.

    Parameters
    ----------
    std : float
        Standard deviation for PDF function.
        If your input features have high values than standard
        deviation should also be high. For instance, if input features
        from range ``[0, 20]`` that standard deviation should be
        also a big value like ``10`` or ``15``. Small values will
        lead to bad prediction.

    {Verbose.verbose}

    Notes
    -----
    - GRNN Network is sensitive for cases when one input feature
      has higher values than the other one. Input data has to be
      normalized before training.

    - Standard deviation has to match the range of the input features
      Check ``std`` parameter description for more information.

    - The bigger training dataset the slower prediction.
      Algorithm is much more efficient for small datasets.

    - Network uses lazy learning which mean that network doesn't
      need iterative training. It just stores parameters
      and use them to make a predictions.

    Methods
    -------
    train(X_train, y_train, copy=True)
        Network just stores all the information about the data and use
        it for the prediction. Parameter ``copy`` copies input data
        before saving it inside the network.

    predict(X)
        Return prediction per each sample in the ``X``.

    {BaseSkeleton.fit}

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import datasets, preprocessing
    >>> from sklearn.model_selection import train_test_split
    >>> from neupy import algorithms
    >>>
    >>> dataset = datasets.load_diabetes()
    >>> x_train, x_test, y_train, y_test = train_test_split(
    ...     preprocessing.minmax_scale(dataset.data),
    ...     preprocessing.minmax_scale(dataset.target.reshape(-1, 1)),
    ...     test_size=0.3,
    ... )
    >>>
    >>> nw = algorithms.GRNN(std=0.1, verbose=False)
    >>> nw.train(x_train, y_train)
    >>>
    >>> y_predicted = nw.predict(x_test)
    >>> mse = np.mean((y_predicted - y_test) ** 2)
    >>> mse
    0.05280970704568171
    """
    std = BoundedProperty(minval=0)

    def __init__(self, std, verbose=False):
        self.std = std
        self.X_train = None
        self.y_train = None
        super(GRNN, self).__init__(verbose=verbose)

    def train(self, X_train, y_train, copy=True):
        """
        Trains network. PNN doesn't actually train, it just stores
        input data and use it for prediction.

        Parameters
        ----------
        X_train : array-like (n_samples, n_features)

        y_train : array-like (n_samples,)
            Target variable should be vector or matrix
            with one feature column.

        copy : bool
            If value equal to ``True`` than input matrices will
            be copied. Defaults to ``True``.

        Raises
        ------
        ValueError
            In case if something is wrong with input data.
        """
        X_train = format_data(X_train, copy=copy)
        y_train = format_data(y_train, copy=copy)

        if y_train.shape[1] != 1:
            raise ValueError("Target value must be one dimensional array")

        self.X_train = X_train
        self.y_train = y_train

        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Number of samples in the input and target "
                             "datasets are different")

    def predict(self, X):
        """
        Make a prediction from the input data.

        Parameters
        ----------
        X : array-like (n_samples, n_features)

        Raises
        ------
        ValueError
            In case if something is wrong with input data.

        Returns
        -------
        array-like (n_samples,)
        """
        if self.X_train is None:
            raise NotTrained(
                "Cannot make a prediction. Network hasn't been trained yet")

        X = format_data(X)

        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                "Input data must contain {0} features, got {1}"
                "".format(self.X_train.shape[1], X.shape[1]))

        ratios = pdf_between_data(self.X_train, X, self.std)
        return (dot(self.y_train.T, ratios) / ratios.sum(axis=0)).T
