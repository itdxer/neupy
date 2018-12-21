import random

import numpy as np
from numpy.core.umath_tests import inner1d

from neupy.utils import format_data
from neupy.exceptions import NotTrained
from .base import DiscreteMemory


__all__ = ('DiscreteBAM',)


def bin2sign(matrix):
    return np.where(matrix == 0, -1, 1)


def sign2bin(matrix):
    return np.where(matrix > 0, 1, 0).astype(int)


def hopfield_energy(weight, X, y):
    return -0.5 * inner1d(X.dot(weight), y)


class DiscreteBAM(DiscreteMemory):
    """
    Discrete BAM Network with associations.
    Network associate every input with some target value.

    Parameters
    ----------
    {DiscreteMemory.Parameters}

    Methods
    -------
    train(X, y)
        Train network and update network weights.

    predict_output(X, n_times=None)
        Using input data recover output data. Returns two arguments.
        First is an input data, second is an output data.

    predict(X, n_times=None)
        Alias to the ``predict_output`` method.

    predict_input(y, n_times=None)
        Using output data recover input data. Returns two arguments.
        First is input data, second is output data.

    energy(X, y)
        Calculate Hopfield Energy for the input and output data.

    Notes
    -----
    - Input and output vectors should contain only binary values.

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
    >>> bamnet = algorithms.DiscreteBAM(mode='sync')
    >>> bamnet.train(data, hints)
    >>>
    >>> recovered_zero, recovered_hint = bamnet.predict(half_zero)
    >>> recovered_hint
    matrix([[0, 1, 0, 0]])
    >>>
    >>> zero_hint
    matrix([[0, 1, 0, 0]])
    """
    def apply_async_process(self, X_sign, y_sign, n_times=None):
        if n_times is None:
            n_times = self.n_times

        n_input_features = X_sign.shape[-1]
        n_output_features = y_sign.shape[-1]

        for _ in range(n_times):
            i = random.randrange(n_input_features)
            j = random.randrange(n_output_features)

            X_sign[:, i] = np.sign(y_sign.dot(self.weight[i, :]))
            y_sign[:, j] = np.sign(X_sign.dot(self.weight[:, j]))

        return sign2bin(X_sign), sign2bin(y_sign)

    def predict_input(self, y_bin, n_times=None):
        if self.weight is None:
            raise NotTrained("Network hasn't been trained yet")

        self.discrete_validation(y_bin)

        y_bin = format_data(y_bin, is_feature1d=False)
        y_sign = bin2sign(y_bin)
        X_sign = np.sign(y_sign.dot(self.weight.T))

        if self.mode == 'sync':
            return sign2bin(X_sign), y_bin

        return self.apply_async_process(X_sign, y_sign, n_times)

    def predict_output(self, X_bin, n_times=None):
        if self.weight is None:
            raise NotTrained("Network hasn't been trained yet")

        self.discrete_validation(X_bin)

        X_bin = format_data(X_bin, is_feature1d=False)
        X_sign = bin2sign(X_bin)
        y_sign = np.sign(X_sign.dot(self.weight))

        if self.mode == 'sync':
            return X_bin, sign2bin(y_sign)

        return self.apply_async_process(X_sign, y_sign, n_times)

    def predict(self, X_bin, n_times=None):
        return self.predict_output(X_bin, n_times)

    def train(self, X_bin, y_bin):
        self.discrete_validation(X_bin)
        self.discrete_validation(y_bin)

        X_sign = bin2sign(format_data(X_bin, is_feature1d=False))
        y_sign = bin2sign(format_data(y_bin, is_feature1d=False))

        _, weight_nrows = X_sign.shape
        _, weight_ncols = y_sign.shape
        weight_shape = (weight_nrows, weight_ncols)

        if self.weight is None:
            self.weight = np.zeros(weight_shape)

        if self.weight.shape != weight_shape:
            raise ValueError(
                "Invalid input shapes. Number of input "
                "features must be equal to {} and {} output "
                "features".format(weight_nrows, weight_ncols))

        self.weight += X_sign.T.dot(y_sign)

    def energy(self, X_bin, y_bin):
        self.discrete_validation(X_bin)
        self.discrete_validation(y_bin)

        X_sign, y_sign = bin2sign(X_bin), bin2sign(y_bin)
        X_sign = format_data(X_sign, is_feature1d=False)
        y_sign = format_data(y_sign, is_feature1d=False)
        nrows, n_features = X_sign.shape

        if nrows == 1:
            return hopfield_energy(self.weight, X_sign, y_sign)

        output = np.zeros(nrows)
        for i, rows in enumerate(zip(X_sign, y_sign)):
            output[i] = hopfield_energy(self.weight, *rows)

        return output
