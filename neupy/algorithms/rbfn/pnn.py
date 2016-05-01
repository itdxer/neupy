from numpy import unique, zeros, dot, sum as np_sum

from neupy.utils import format_data
from neupy.core.properties import BoundedProperty
from neupy.network.base import BaseNetwork
from neupy.network.learning import LazyLearning

from .utils import pdf_between_data


__all__ = ('PNN',)


class PNN(LazyLearning, BaseNetwork):
    """ Probabilistic Neural Network for classification.

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
    >>> import numpy as np
    >>>
    >>> from sklearn import datasets
    >>> from sklearn import metrics
    >>> from sklearn.cross_validation import train_test_split
    >>> from neupy import algorithms, environment
    >>>
    >>> environment.reproducible()
    >>>
    >>> dataset = datasets.load_digits()
    >>> x_train, x_test, y_train, y_test = train_test_split(
    ...     dataset.data, dataset.target, train_size=0.7
    ... )
    >>>
    >>> nw = algorithms.PNN(std=10, verbose=False)
    >>> nw.train(x_train, y_train)
    >>> result = nw.predict(x_test)
    >>> metrics.accuracy_score(y_test, result)
    0.98888888888888893
    """
    std = BoundedProperty(default=0.1, minval=0)

    def __init__(self, **options):
        super(PNN, self).__init__(**options)
        self.classes = None

    def train(self, input_train, target_train, copy=True):
        input_train = format_data(input_train, copy=copy)
        target_train = format_data(target_train, copy=copy)

        LazyLearning.train(self, input_train, target_train)

        if target_train.shape[1] != 1:
            raise ValueError("Target value must be in 1 dimention")

        classes = self.classes = unique(target_train)
        number_of_classes = classes.size
        row_comb_matrix = self.row_comb_matrix = zeros(
            (number_of_classes, input_train.shape[0])
        )
        class_ratios = self.class_ratios = zeros(number_of_classes)

        for i, class_name in enumerate(classes):
            class_val_positions = (target_train == i)
            row_comb_matrix[i, class_val_positions.ravel()] = 1
            class_ratios[i] = np_sum(class_val_positions)

    def predict_proba(self, input_data):
        raw_output = self.predict_raw(input_data)

        total_output_sum = raw_output.sum(axis=0).reshape(
            (raw_output.shape[1], 1)
        )
        return raw_output.T / total_output_sum

    def predict_raw(self, input_data):
        input_data = format_data(input_data)
        super(PNN, self).predict(input_data)

        if self.classes is None:
            raise ValueError("Train network before predict data")

        input_data_size = input_data.shape[1]
        train_data_size = self.input_train.shape[1]

        if input_data_size != train_data_size:
            raise ValueError("Input data must contains {0} features, got "
                             "{1}".format(train_data_size, input_data_size))

        class_ratios = self.class_ratios
        pdf_outputs = pdf_between_data(self.input_train, input_data,
                                       self.std)
        return dot(
            self.row_comb_matrix, pdf_outputs
        ) / class_ratios.reshape((class_ratios.size, 1))

    def predict(self, input_data):
        raw_output = self.predict_raw(input_data)
        return self.classes[raw_output.argmax(axis=0)]
