from numpy import dot

from neuralpy.core.properties import NonNegativeNumberProperty
from neuralpy.network.base import BaseNetwork
from neuralpy.network.connections import FAKE_CONNECTION
from neuralpy.network.learning import LazyLearning
from neuralpy.network.types import Regression
from .utils import pdf_between_data


__all__ = ('GRNN',)


class GRNN(LazyLearning, Regression, BaseNetwork):
    """ Generalized Regression Neural Network.

    Parameters
    ----------
    standard_deviation : float
        standard deviation for PDF function, default to 0.1.
    {full_params}

    Methods
    -------
    {supervised_train_lazy}
    {full_methods}

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> from sklearn import datasets
    >>> from sklearn.cross_validation import train_test_split
    >>>
    >>> from neuralpy.algorithms import GRNN
    >>> from neuralpy.functions import rmsle
    >>>
    >>> np.random.seed(0)
    >>>
    >>> dataset = datasets.load_diabetes()
    >>> x_train, x_test, y_train, y_test = train_test_split(
    ...     dataset.data, dataset.target, train_size=0.7,
    ...     random_state=0
    ... )
    >>>
    >>> nw = GRNN(standard_deviation=0.1, verbose=False)
    >>> nw.train(x_train, y_train)
    >>> result = nw.predict(x_test)
    >>> rmsle(result, y_test)
    0.4245120142774001
    """
    standard_deviation = NonNegativeNumberProperty(default=0.1)

    def __init__(self, **options):
        super(GRNN, self).__init__(FAKE_CONNECTION, **options)

    def train(self, input_train, target_train):
        if target_train.ndim != 1:
            raise ValueError("Target value must be in 1 dimention")
        LazyLearning.train(self, input_train, target_train)

    def predict(self, input_data):
        super(GRNN, self).predict(input_data)

        input_data_size = input_data.shape[1]
        train_data_size = self.input_train.shape[1]

        if input_data_size != train_data_size:
            raise ValueError("Input data must contains {0} features, got "
                             "{1}".format(train_data_size, input_data_size))

        ratios = pdf_between_data(self.input_train, input_data,
                                  self.standard_deviation)
        return dot(ratios.T, self.target_train) / ratios.sum(axis=0)
