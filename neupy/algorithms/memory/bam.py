from random import randint

from numpy import zeros, sign

from neupy.utils import format_data
from neupy.functions import step
from .utils import bin2sign, hopfield_energy
from .base import DiscreteMemory


__all__ = ('DiscreteBAM',)


class DiscreteBAM(DiscreteMemory):
    """ Discrete BAM Network with associations.
    Network associate every input with some target value.

    Parameters
    ----------
    {discrete_params}

    Methods
    -------
    train(input_data, output_data)
        Train network and update network weights.
    predict_output(input_data, n_times=None)
        Using input data recover output data. Returns two arguments.
        First is input data, second is output data.
    predict(input_data, n_times=None)
        The same as ``predict_output``.
    predict_input(output_data, n_times=None)
        Using output data recover input data. Returns two arguments.
        First is input data, second is output data.
    energy(input_data, output_data)
        Calculate Hopfiel Energy for the input and output data.

    Notes
    -----
    * {discrete_data_note}

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
    >>> zero_hint
    matrix([[0, 1, 0, 0]])
    """

    def format_predict(self, predicted_result):
        return step(predicted_result).astype(int)

    def predict_input(self, output_data, n_times=None):
        return self._predict(input_data=None,
                             output_data=format_data(output_data, row1d=True),
                             n_times=n_times)

    def predict_output(self, input_data, n_times=None):
        return self._predict(input_data=format_data(input_data, row1d=True),
                             output_data=None,
                             n_times=n_times)

    def _predict(self, input_data=None, output_data=None, n_times=None):
        if self.weight is None:
            raise AttributeError("Train network before predict the values")

        if input_data is None and output_data is not None:
            self.discrete_validation(output_data)
            output_data = bin2sign(output_data)
            input_data = sign(output_data.dot(self.weight.T))

        elif input_data is not None and output_data is None:
            self.discrete_validation(input_data)
            input_data = bin2sign(input_data)
            output_data = sign(input_data.dot(self.weight))

        else:
            raise ValueError("Input or output data have to be equal to `None`")

        n_output_features = output_data.shape[-1]
        n_input_features = input_data.shape[-1]

        if self.mode == 'async':
            if n_times is None:
                n_times = self.n_times

            for _ in range(n_times):
                input_position = randint(0, n_input_features - 1)
                output_position = randint(0, n_output_features - 1)

                input_data[:, input_position] = sign(
                    output_data.dot(self.weight[input_position, :])
                )
                output_data[:, output_position] = sign(
                    input_data.dot(self.weight[:, output_position])
                )

        return (
            self.format_predict(input_data),
            self.format_predict(output_data),
        )

    def train(self, input_data, output_data):
        self.discrete_validation(input_data)
        self.discrete_validation(output_data)

        output_data = bin2sign(format_data(output_data, row1d=True))
        input_data = bin2sign(format_data(input_data, row1d=True))

        _, wight_nrows = input_data.shape
        _, wight_ncols = output_data.shape
        weight_shape = (wight_nrows, wight_ncols)

        if self.weight is None:
            self.weight = zeros(weight_shape)

        if self.weight.shape != weight_shape:
            raise ValueError("Invalid input shapes. Number of input "
                             "features must be equal to {} and {} output "
                             "features".format(wight_nrows, wight_ncols))

        self.weight += input_data.T.dot(output_data)

    def energy(self, input_data, output_data):
        self.discrete_validation(input_data)
        self.discrete_validation(output_data)

        input_data, output_data = bin2sign(input_data), bin2sign(output_data)
        input_data = format_data(input_data, row1d=True)
        output_data = format_data(output_data, row1d=True)
        nrows, n_features = input_data.shape

        if nrows == 1:
            return hopfield_energy(self.weight, input_data, output_data)

        output = zeros(nrows)
        for i, rows in enumerate(zip(input_data, output_data)):
            output[i] = hopfield_energy(self.weight, *rows)

        return output

    predict = predict_output
