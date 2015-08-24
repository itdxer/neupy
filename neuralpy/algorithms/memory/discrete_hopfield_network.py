from math import log, ceil

from numpy import zeros, fill_diagonal, random, multiply

from neuralpy.core.properties import ChoiceProperty, NonNegativeIntProperty
from neuralpy.functions import signum
from .utils import bin2sign
from .base import DiscreteMemory


__all__ = ('DiscreteHopfieldNetwork',)


class DiscreteHopfieldNetwork(DiscreteMemory):
    """ Discrete Hopfield Network. Memory algorithm which works only with
    binary vectors.

    Notes
    -----
    * {discrete_data_note}

    Examples
    --------
    >>> import numpy as np
    >>> from neuralpy import algorithms
    >>>
    >>> def draw_bin_image(image_matrix):
    ...     for row in image_matrix.tolist():
    ...         print('| ' + ' '.join(' *'[val] for val in row))
    ...
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
    >>>
    >>> two = np.matrix([
    ...     1, 1, 1, 0, 0,
    ...     0, 0, 0, 1, 0,
    ...     0, 0, 0, 1, 0,
    ...     0, 1, 1, 0, 0,
    ...     1, 0, 0, 0, 0,
    ...     1, 1, 1, 1, 1,
    ... ])
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
    >>> draw_bin_image(zero.reshape((6, 5)))
    |   * * *
    | *       *
    | *       *
    | *       *
    | *       *
    |   * * *
    >>> draw_bin_image(half_zero.reshape((6, 5)))
    |   * * *
    | *       *
    | *       *
    |
    |
    |
    >>> data = np.concatenate([zero, one, two], axis=0)
    >>>
    >>> dhnet = algorithms.DiscreteHopfieldNetwork()
    >>> dhnet.train(data)
    >>>
    >>> result = dhnet.predict(half_zero)
    >>> draw_bin_image(result.reshape((6, 5)))
    |   * * *
    | *       *
    | *       *
    | *       *
    | *       *
    |   * * *
    """
    mode = ChoiceProperty(default='full', choices=['random', 'full'])
    n_nodes = NonNegativeIntProperty(default=100)

    def __init__(self, **options):
        super(DiscreteHopfieldNetwork, self).__init__(**options)
        self.n_remembered_data = 0
        self.weight = None

        if 'n_nodes' in options and self.mode != 'random':
            self.logs.warning("You can use `n_nodes` property only in "
                              "`random` mode.")

    def train(self, input_data):
        self.discrete_validation(input_data)

        input_data = bin2sign(input_data)

        nrows, n_features = input_data.shape
        nrows_after_update = self.n_remembered_data + nrows
        memory_limit = ceil(n_features / (2 * log(n_features)))

        if nrows_after_update > memory_limit:
            raise ValueError("You can't memorize more than {0} "
                             "samples".format(memory_limit))

        weight_shape = (n_features, n_features)

        if self.weight is None:
            self.weight = zeros(weight_shape)

        if self.weight.shape != weight_shape:
            raise ValueError("Invalid input shapes. Number of input "
                             "features must be equal to {} and output "
                             "features - {}".format(*weight_shape))

        self.weight = input_data.T.dot(input_data)
        fill_diagonal(self.weight, zeros(len(self.weight)))
        self.n_remembered_data = nrows_after_update

    def predict(self, input_data):
        input_data = bin2sign(input_data)

        if self.mode == 'random':
            # Valid number of features for one or two dimentions
            n_features = input_data.shape[-1]

            if input_data.ndim == 1:
                input_data = input_data.reshape((1, n_features))

            data = zeros(input_data.shape)
            for _ in range(self.n_nodes):
                data[:, random.randint(0, n_features - 1)] += 1

            input_data = multiply(input_data, data)

        predicted = signum(input_data.dot(self.weight),
                           upper_value=1, lower_value=0)
        return predicted.astype(int)
