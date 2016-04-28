from __future__ import division

from numpy import dot, zeros, ones, inf, logical_and, sort, unique

from neupy.utils import format_data
from neupy.core.properties import (ProperFractionProperty,
                                   IntProperty)
from neupy.network.base import BaseNetwork


__all__ = ('ART1',)


class ART1(BaseNetwork):
    """ Adaptive Resonance Theory (ART1) Network for binary
    data clustering.

    Notes
    -----
    * Weights are not random, so the result will be always reproduceble.

    Parameters
    ----------
    rho : float
        Control reset action in training process. Value must be
        between ``0`` and ``1``, defaults to ``0.5``.
    n_clusters : int
        Number of clusters, defaults to ``2``. Min value is also ``2``.
    {BaseNetwork.step}
    {BaseNetwork.show_epoch}
    {BaseNetwork.shuffle_data}
    {BaseNetwork.epoch_end_signal}
    {BaseNetwork.train_end_signal}

    Methods
    -------
    train(input_data):
        Network network will train until it clusters all samples.
    {BaseSkeleton.predict}
    {BaseSkeleton.fit}

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms
    >>>
    >>> data = np.array([
    ...     [0, 1, 0],
    ...     [1, 0, 0],
    ...     [1, 1, 0],
    ... ])
    >>>>
    >>> artnet = algorithms.ART1(
    ...     step=2,
    ...     rho=0.7,
    ...     n_clusters=2,
    ...     verbose=False
    ... )
    >>> artnet.predict(data)
    array([ 0.,  1.,  1.])
    """
    rho = ProperFractionProperty(default=0.5)
    n_clusters = IntProperty(default=2, minval=2)

    def train(self, input_data):
        input_data = format_data(input_data)

        if input_data.ndim != 2:
            raise ValueError("Input value must be 2 dimentional, got "
                             "{0}".format(input_data.ndim))

        data_size = input_data.shape[1]
        n_clusters = self.n_clusters
        step = self.step
        rho = self.rho

        if list(sort(unique(input_data))) != [0, 1]:
            raise ValueError("ART1 Network works only with binary matrix, "
                             "all matix must contains only 0 and 1")

        if not hasattr(self, 'weight_21'):
            self.weight_21 = ones((data_size, n_clusters))

        if not hasattr(self, 'weight_12'):
            self.weight_12 = step / (step + n_clusters - 1) * self.weight_21.T

        weight_21 = self.weight_21
        weight_12 = self.weight_12

        if data_size != weight_21.shape[0]:
            raise ValueError(
                "Data dimention is invalid. Get {} columns data set. "
                "Must be - {} columns".format(
                    data_size, weight_21.shape[0]
                )
            )

        classes = zeros(input_data.shape[0])

        # Train network
        for i, p in enumerate(input_data):
            disabled_neurons = []
            reseted_values = []
            reset = True

            while reset:
                output1 = p
                input2 = dot(weight_12, output1.T)

                output2 = zeros(input2.size)
                input2[disabled_neurons] = -inf
                winner_index = input2.argmax()
                output2[winner_index] = 1

                expectation = dot(weight_21, output2)
                output1 = logical_and(p, expectation).astype(int)

                reset_value = dot(output1.T, output1) / dot(p.T, p)
                reset = reset_value < rho

                if reset:
                    disabled_neurons.append(winner_index)
                    reseted_values.append((reset_value, winner_index))

                if len(disabled_neurons) >= n_clusters:
                    # Got this case only if we test all possible clusters
                    reset = False
                    winner_index = None

                if not reset:
                    if winner_index is not None:
                        weight_12[winner_index, :] = (step * output1) / (
                            step + dot(output1.T, output1) - 1
                        )
                        weight_21[:, winner_index] = output1
                    else:
                        # Get result with the best `rho`
                        winner_index = max(reseted_values)[1]

                    classes[i] = winner_index

        return classes

    def predict(self, input_data):
        return self.train(input_data)
