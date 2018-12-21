import math
import random

import numpy as np
from numpy.core.umath_tests import inner1d

from neupy.utils import format_data
from neupy.core.properties import Property
from .base import DiscreteMemory


__all__ = ('DiscreteHopfieldNetwork',)


def bin2sign(matrix):
    return np.where(matrix == 0, -1, 1)


def hopfield_energy(weight, X, output_data):
    return -0.5 * inner1d(X.dot(weight), output_data)


class DiscreteHopfieldNetwork(DiscreteMemory):
    """
    Discrete Hopfield Network. It can memorize binary samples
    and reconstruct them from corrupted samples.

    Notes
    -----
    - Works only with binary data. Input matrix should
      contain only zeros and ones.

    Parameters
    ----------
    {DiscreteMemory.Parameters}

    check_limit : bool
        Option enable a limit of patterns control for the
        network using logarithmically proportion rule.
        Defaults to ``True``.

        .. math::

            \\frac{{n_{{features}}}}{{2 \\cdot log_{{e}}(n_{{features}})}}

    Methods
    -------
    energy(X)
        Computes Discrete Hopfield Energy.

    train(X)
        Save input data pattern into the network's memory. Each call will
        make partial fit for the network.

    predict(X, n_times=None)
        Recover data from the memory using input pattern.
        For the prediction procedure you can control number
        of iterations. If you set up this value equal to ``None``
        then the value would be equal to the value that you
        set up for the property with the same name - ``n_times``.

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
    :ref:`discrete-hopfield-network`: Discrete Hopfield Network article.
    """
    check_limit = Property(expected_type=bool)

    def __init__(self, mode='sync', n_times=100, verbose=False,
                 check_limit=True):

        self.n_memorized_samples = 0
        self.check_limit = check_limit

        super(DiscreteHopfieldNetwork, self).__init__(
            mode=mode, n_times=n_times, verbose=verbose)

    def train(self, X_bin):
        self.discrete_validation(X_bin)

        X_sign = bin2sign(X_bin)
        X_sign = format_data(X_sign, is_feature1d=False, make_float=False)

        n_rows, n_features = X_sign.shape
        n_rows_after_update = self.n_memorized_samples + n_rows

        if self.check_limit:
            memory_limit = math.ceil(n_features / (2 * math.log(n_features)))

            if n_rows_after_update > memory_limit:
                raise ValueError("You can't memorize more than {0} "
                                 "samples".format(memory_limit))

        weight_shape = (n_features, n_features)

        if self.weight is None:
            self.weight = np.zeros(weight_shape, dtype=int)

        if self.weight.shape != weight_shape:
            n_features_expected = self.weight.shape[1]
            raise ValueError("Input data has invalid number of features. "
                             "Got {} features instead of {}."
                             "".format(n_features, n_features_expected))

        self.weight += X_sign.T.dot(X_sign)
        np.fill_diagonal(self.weight, np.zeros(len(self.weight)))
        self.n_memorized_samples = n_rows_after_update

    def predict(self, X_bin, n_times=None):
        self.discrete_validation(X_bin)

        X_sign = bin2sign(X_bin)
        X_sign = format_data(X_sign, is_feature1d=False, make_float=False)

        if self.mode == 'async':
            _, n_features = X_sign.shape

            if n_times is None:
                n_times = self.n_times

            for _ in range(n_times):
                position = random.randrange(n_features)
                raw_new_value = X_sign.dot(self.weight[:, position])
                X_sign[:, position] = np.sign(raw_new_value)
        else:
            X_sign = X_sign.dot(self.weight)

        return np.where(X_sign > 0, 1, 0).astype(int)

    def energy(self, X_bin):
        self.discrete_validation(X_bin)

        X_sign = bin2sign(X_bin)
        X_sign = format_data(X_sign, is_feature1d=False, make_float=False)

        n_rows, n_features = X_sign.shape

        if n_rows == 1:
            return hopfield_energy(self.weight, X_sign, X_sign)

        output = np.zeros(n_rows)
        for i, row in enumerate(X_sign):
            output[i] = hopfield_energy(self.weight, row, row)

        return output
