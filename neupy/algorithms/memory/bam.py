from numpy import zeros, sign

from .utils import sign2bin, bin2sign, format_data, hopfield_energy
from .base import DiscreteMemory


__all__ = ('DiscreteBAM',)


class DiscreteBAM(DiscreteMemory):
    """ Discrete BAM Network with associations.
    Network associate every input with some target value.

    Notes
    -----
    *{discrete_data_note}

    Methods
    -------
    train(input_data, output_data)
        Train network and update network weights.
    predict_output(input_data)
        Based on input date network predict it output.
    predict(input_data)
        The same as ``predict_output``
    predict_input(output_data)
        Based on output date network predict it input.
    energy(input_data, output_data)
        Compute Discrete Hopfiel Energy.

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms
    >>>
    >>> zero = np.matrix([
    ...     0, 1, 1, 1, 0,
    ...     1, 0, 0, 0, 1,
    ...     1, 0, 0, 0, 1,
    ...     1, 0, 0, 0, 1,
    ...     1, 0, 0, 0, 1,
    ...     0, 1, 1, 1, 0
    ... ])
    >>>
    >>> one = np.matrix([
    ...     0, 1, 1, 0, 0,
    ...     0, 0, 1, 0, 0,
    ...     0, 0, 1, 0, 0,
    ...     0, 0, 1, 0, 0,
    ...     0, 0, 1, 0, 0,
    ...     0, 0, 1, 0, 0
    ... ])
    >>> zero.reshape((5, 6))
    >>>
    >>> half_zero = np.matrix([
    ...     0, 1, 1, 1, 0,
    ...     1, 0, 0, 0, 1,
    ...     1, 0, 0, 0, 1,
    ...     0, 0, 0, 0, 0,
    ...     0, 0, 0, 0, 0,
    ...     0, 0, 0, 0, 0,
    ... ])
    >>>
    >>> zero_hint = np.matrix([[0, 1, 0, 0]])
    >>> one_hint = np.matrix([[1, 0, 0, 0]])
    >>>
    >>> data = np.concatenate([zero, one], axis=0)
    >>> hints = np.concatenate([zero_hint, one_hint], axis=0)
    >>>
    >>> bamnet = algorithms.DiscreteBAM()
    >>> bamnet.train(data.copy(), hints.copy())
    >>>
    >>> bamnet.predict(half_zero)
    matrix([[0, 1, 0, 0]])
    >>> zero_hint
    matrix([[0, 1, 0, 0]])
    """
    def __init__(self, **options):
        super(DiscreteBAM, self).__init__(**options)
        self.weight = None

    def format_predict(self, predicted_result):
        if self.weight is None:
            raise AttributeError("Train the network before predict the values")
        return sign2bin(sign(predicted_result).astype(int))

    def predict_input(self, output_data):
        output_data = bin2sign(output_data)
        return self.format_predict(output_data.dot(self.weight.T))

    def predict_output(self, input_data):
        input_data = bin2sign(input_data)
        return self.format_predict(input_data.dot(self.weight))

    def train(self, input_data, output_data):
        self.discrete_validation(input_data)
        self.discrete_validation(output_data)

        output_data = bin2sign(output_data)
        input_data = bin2sign(input_data)

        _, wight_nrows = input_data.shape
        _, wight_ncols = output_data.shape
        weight_shape = (wight_nrows, wight_ncols)

        if self.weight is None:
            self.weight = zeros(weight_shape)

        if self.weight.shape != weight_shape:
            raise ValueError("Invalid input shapes. Number of input "
                             "features must be equal to {} and output "
                             "features - {}".format(wight_nrows, wight_ncols))

        self.weight += input_data.T.dot(output_data)

    def energy(self, input_data, output_data):
        input_data, output_data = bin2sign(input_data), bin2sign(output_data)
        input_data = format_data(input_data)
        output_data = format_data(output_data)
        nrows, n_features = input_data.shape

        if nrows == 1:
            return hopfield_energy(self.weight, input_data, output_data)

        output = zeros(nrows)
        for i, rows in enumerate(zip(input_data, output_data)):
            output[i] = hopfield_energy(self.weight, *rows)

        return output

    predict = predict_output
