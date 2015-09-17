from math import log, ceil

from numpy import zeros, fill_diagonal, random, multiply

from neupy.core.properties import (ChoiceProperty, NonNegativeIntProperty,
                                   BoolProperty)
from neupy.functions import signum
from .utils import bin2sign, hopfield_energy, format_data
from .base import DiscreteMemory


__all__ = ('DiscreteHopfieldNetwork',)


class DiscreteHopfieldNetwork(DiscreteMemory):
    """ Discrete Hopfield Network. Memory algorithm which works only with
    binary vectors.

    Parameters
    ----------
    mode : {{'sync', 'async'}}
        Indentify pattern recovery mode. ``sync`` mode try recovery a pattern
        using the all input vector. ``async`` mode randomly chose some
        values from the input vector and repeat this procedure the number
        of times a given variable ``n_times``. Defaults to ``sync``.
    n_times : int
        Available only in ``async`` mode. Identify number of random trials.
        Defaults to ``100``.
    check_limit : bool
        Option enable a limit of patterns control for the network using
        logarithmically proportion rule. Defaults to ``True``.

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
    mode = ChoiceProperty(default='sync', choices=['async', 'sync'])
    n_times = NonNegativeIntProperty(default=100)
    check_limit = BoolProperty(default=True)

    def __init__(self, **options):
        super(DiscreteHopfieldNetwork, self).__init__(**options)
        self.n_remembered_data = 0
        self.weight = None

        if 'n_times' in options and self.mode != 'async':
            self.logs.warning("You can use `n_times` property only in "
                              "`async` mode.")

    def train(self, input_data):
        self.discrete_validation(input_data)

        input_data = bin2sign(input_data)
        input_data = format_data(input_data)

        nrows, n_features = input_data.shape
        nrows_after_update = self.n_remembered_data + nrows

        if self.check_limit:
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

        if self.mode == 'async':
            input_data = format_data(input_data)
            _, n_features = input_data.shape

            data = zeros(input_data.shape)
            for _ in range(self.n_times):
                data[:, random.randint(0, n_features - 1)] += 1

            input_data = multiply(input_data, data)

        predicted = signum(input_data.dot(self.weight),
                           lower_value=0, upper_value=1)
        return predicted.astype(int)

    def energy(self, input_data):
        input_data = bin2sign(input_data)
        input_data = format_data(input_data)
        nrows, n_features = input_data.shape

        if nrows == 1:
            return hopfield_energy(self.weight, input_data, input_data)

        output = zeros(nrows)
        for i, row in enumerate(input_data):
            output[i] = hopfield_energy(self.weight, row, row)

        return output
