from math import log, ceil

from numpy import zeros, fill_diagonal, random, multiply

from neupy.core.properties import ChoiceProperty, NonNegativeIntProperty
from neupy.functions import signum
from .utils import bin2sign
from .base import DiscreteMemory


__all__ = ('DiscreteHopfieldNetwork',)


def format_data(input_data):
    # Valid number of features for one or two dimentions
    n_features = input_data.shape[-1]
    if input_data.ndim == 1:
        input_data = input_data.reshape((1, n_features))
    return input_data


class DiscreteHopfieldNetwork(DiscreteMemory):
    """ Discrete Hopfield Network. Memory algorithm which works only with
    binary vectors.

    Parameters
    ----------
    mode : {{'full', 'random'}}
        Indentify pattern recovery mode. ``full`` mode try recovery a pattern
        using the all input vector. ``random`` mode randomly chose some
        values from the input vector and repeat this procedure the number
        of times a given variable ``n_nodes``. Defaults to ``full``.
    n_nodes : int
        Available only in ``random`` mode. Identify number of random trials.
        Defaults to ``100``.

    Methods
    -------
    energy(input_data)
        Compute Discrete Hopfiel Energy.
    train(input_data)
        Save input data pattern into the network memory.
    predict(input_data)
        Recover data from the memory using input pattern.

    Notes
    -----
    * {discrete_data_note}

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms
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

    See Also
    --------
    :ref:`password-recovery`: Password recovery with Discrete Hopfield Network.
    :ref:`discrete-hopfield-network`: Discrete Hopfield Network tutorial.
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
        input_data = format_data(input_data)

        nrows, n_features = input_data.shape
        nrows_after_update = self.n_remembered_data + nrows
        memory_limit = ceil(n_features / (2 * log(n_features)))

        if nrows_after_update > memory_limit:
            raise ValueError("You can't memorize more than {0} "
                             "samples".format(memory_limit))

        weight_shape = (n_features, n_features)

        if self.weight is None:
            self.weight = zeros(weight_shape, dtype=int)

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
            input_data = format_data(input_data)
            _, n_features = input_data.shape

            data = zeros(input_data.shape)
            for _ in range(self.n_nodes):
                data[:, random.randint(0, n_features - 1)] += 1

            input_data = multiply(input_data, data)

        predicted = signum(input_data.dot(self.weight),
                           lower_value=0, upper_value=1)
        return predicted.astype(int)

    def _hopfield_energy(self, input_data):
        energy_output = -0.5 * input_data.dot(self.weight).dot(input_data.T)
        return energy_output.item(0)

    def energy(self, input_data):
        input_data = bin2sign(input_data)
        input_data = format_data(input_data)
        nrows, n_features = input_data.shape

        if nrows == 1:
            return self._hopfield_energy(input_data)

        output = zeros(nrows)
        for i, row in enumerate(input_data):
            output[i] = self._hopfield_energy(row)

        return output
