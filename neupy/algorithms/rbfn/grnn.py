from numpy import dot

from neupy.utils import format_data
from neupy.core.properties import BoundedProperty
from neupy.network.base import BaseNetwork
from neupy.network.learning import LazyLearning
from .utils import pdf_between_data


__all__ = ('GRNN',)


class GRNN(LazyLearning, BaseNetwork):
    """ Generalized Regression Neural Network.

    Parameters
    ----------
    std : float
        standard deviation for PDF function, default to 0.1.
    {Verbose.verbose}

    Methods
    -------
    {LazyLearning.train}
    {BaseSkeleton.predict}
    {BaseSkeleton.fit}

    Examples
    --------
    >>> from sklearn import datasets
    >>> from sklearn.cross_validation import train_test_split
    >>> from neupy import algorithms, estimators, environment
    >>>
    >>> environment.reproducible()
    >>>
    >>> dataset = datasets.load_diabetes()
    >>> x_train, x_test, y_train, y_test = train_test_split(
    ...     dataset.data, dataset.target, train_size=0.7,
    ...     random_state=0
    ... )
    >>>
    >>> nw = algorithms.GRNN(std=0.1, verbose=False)
    >>> nw.train(x_train, y_train)
    >>> result = nw.predict(x_test)
    >>> estimators.rmsle(result, y_test)
    0.4245120142774001
    """
    std = BoundedProperty(default=0.1, minval=0)

    def train(self, input_train, target_train, copy=True):
        input_train = format_data(input_train, copy=copy)
        target_train = format_data(target_train, copy=copy)

        if target_train.shape[1] != 1:
            raise ValueError("Target value must be one dimentional array")

        LazyLearning.train(self, input_train, target_train)

    def predict(self, input_data):
        super(GRNN, self).predict(input_data)

        input_data = format_data(input_data)

        input_data_size = input_data.shape[1]
        train_data_size = self.input_train.shape[1]

        if input_data_size != train_data_size:
            raise ValueError("Input data must contains {0} features, got "
                             "{1}".format(train_data_size, input_data_size))

        ratios = pdf_between_data(self.input_train, input_data,
                                  self.std)
        return (dot(self.target_train.T, ratios) / ratios.sum(axis=0)).T
