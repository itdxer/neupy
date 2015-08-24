from numpy import unique, zeros, dot, sum as np_sum

from neuralpy import layers
from neuralpy.core.properties import NonNegativeNumberProperty
from neuralpy.network.base import BaseNetwork
from neuralpy.network.connections import FAKE_CONNECTION
from neuralpy.network.learning import LazyLearning
from neuralpy.network.types import Classification

from .utils import pdf_between_data


__all__ = ('PNN',)


class PNN(LazyLearning, Classification, BaseNetwork):
    """ Probabilistic Neural Network for classification.

    Parameters
    ----------
    standard_deviation : float
        standard deviation for PDF function, default to 0.1.
    {show_epoch}
    {shuffle_data}
    {full_signals}
    {verbose}

    Methods
    -------
    {supervised_train_lazy}
    {full_methods}

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> from sklearn import datasets
    >>> from sklearn import metrics
    >>> from sklearn.cross_validation import train_test_split
    >>>
    >>> from neuralpy.algorithms import PNN
    >>>
    >>> np.random.seed(0)
    >>>
    >>> dataset = datasets.load_digits()
    >>> x_train, x_test, y_train, y_test = train_test_split(
    ...     dataset.data, dataset.target, train_size=0.7
    ... )
    >>>
    >>> nw = PNN(standard_deviation=10, verbose=False)
    >>> nw.train(x_train, y_train)
    >>> result = nw.predict(x_test)
    >>> metrics.accuracy_score(y_test, result)
    0.98888888888888893
    """
    standard_deviation = NonNegativeNumberProperty(default=0.1)

    def __init__(self, **options):
        super(PNN, self).__init__(FAKE_CONNECTION, **options)
        self.output_layer = layers.OutputLayer(1)
        self.classes = None

    def train(self, input_train, target_train):
        LazyLearning.train(self, input_train, target_train)

        if target_train.ndim != 1:
            raise ValueError("Target value must be in 1 dimention")

        classes = self.classes = unique(target_train)
        number_of_classes = classes.size
        row_comb_matrix = self.row_comb_matrix = zeros(
            (number_of_classes, input_train.shape[0])
        )
        class_ratios = self.class_ratios = zeros(number_of_classes)

        for i, class_name in enumerate(classes):
            class_val_positions = (target_train == i)
            row_comb_matrix[i, class_val_positions] = 1
            class_ratios[i] = np_sum(class_val_positions)

    def setup_defaults(self):
        # Remove properties from BaseNetwork
        del self.use_bias
        del self.error
        del self.step
        super(PNN, self).setup_defaults()

    def predict_prob(self, input_data):
        raw_output = self.raw_predict(input_data)
        total_output_sum = raw_output.sum(axis=0).reshape(
            (raw_output.shape[1], 1)
        )
        return raw_output.T / total_output_sum

    def raw_predict(self, input_data):
        super(PNN, self).predict(input_data)

        if self.classes is None:
            raise ValueError("Train network before get prediction")

        input_data_size = input_data.shape[1]
        train_data_size = self.input_train.shape[1]

        if input_data_size != train_data_size:
            raise ValueError("Input data must contains {0} features, got "
                             "{1}".format(train_data_size, input_data_size))

        class_ratios = self.class_ratios
        pdf_outputs = pdf_between_data(self.input_train, input_data,
                                       self.standard_deviation)
        return dot(
            self.row_comb_matrix, pdf_outputs
        ) / class_ratios.reshape((class_ratios.size, 1))

    def predict(self, input_data):
        raw_output = self.raw_predict(input_data)
        return self.classes[raw_output.argmax(axis=0)]
