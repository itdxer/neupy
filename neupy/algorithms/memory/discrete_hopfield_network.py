from math import log, ceil

from numpy import zeros, fill_diagonal, random, sign

from neupy.utils import format_data
from neupy.functions import step
from neupy.core.properties import BoolProperty
from .utils import bin2sign, hopfield_energy
from .base import DiscreteMemory


__all__ = ('DiscreteHopfieldNetwork',)


class DiscreteHopfieldNetwork(DiscreteMemory):
    """ Discrete Hopfield Network. Memory algorithm which works only with
    binary vectors.

    Parameters
    ----------
    {discrete_params}
    check_limit : bool
        Option enable a limit of patterns control for the network using
        logarithmically proportion rule. Defaults to ``True``.

    Methods
    -------
    energy(input_data)
        Compute Discrete Hopfiel Energy.
    train(input_data)
        Save input data pattern into the network memory.
    predict(input_data, n_times=None)
        Recover data from the memory using input pattern.
        For the prediction procedure you can control number of iterations.
        If you set up this value equal to ``None`` then the value would be
        equal to the value that you set up for the property with
        the same name - ``n_times``.

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
    check_limit = BoolProperty(default=True)

    def __init__(self, **options):
        super(DiscreteHopfieldNetwork, self).__init__(**options)
        self.n_remembered_data = 0

    def train(self, input_data):
        self.discrete_validation(input_data)

        input_data = bin2sign(input_data)
        input_data = format_data(input_data, row1d=True)

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
                             "features must be equal to {} and {} output "
                             "features".format(*weight_shape))

        self.weight = input_data.T.dot(input_data)
        fill_diagonal(self.weight, zeros(len(self.weight)))
        self.n_remembered_data = nrows_after_update

    def predict(self, input_data, n_times=None):
        self.discrete_validation(input_data)
        input_data = format_data(bin2sign(input_data), row1d=True)

        if self.mode == 'async':
            if n_times is None:
                n_times = self.n_times

            _, n_features = input_data.shape
            output_data = input_data

            for _ in range(n_times):
                position = random.randint(0, n_features - 1)
                raw_new_value = output_data.dot(self.weight[:, position])
                output_data[:, position] = sign(raw_new_value)
        else:
            output_data = input_data.dot(self.weight)

        return step(output_data).astype(int)

    def energy(self, input_data):
        self.discrete_validation(input_data)
        input_data = bin2sign(input_data)
        input_data = format_data(input_data, row1d=True)
        nrows, n_features = input_data.shape

        if nrows == 1:
            return hopfield_energy(self.weight, input_data, input_data)

        output = zeros(nrows)
        for i, row in enumerate(input_data):
            output[i] = hopfield_energy(self.weight, row, row)

        return output
