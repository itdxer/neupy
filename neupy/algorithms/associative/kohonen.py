from numpy import reshape, nonzero

from neupy.utils import format_data
from .base import BaseAssociative


__all__ = ('Kohonen',)


class Kohonen(BaseAssociative):
    """Kohonen unsupervised associative Neural Network.
    This algorith similar to :network:`Instar`. Like the instar rule, the
    Kohonen rule allows the weights of a neuron to learn an input vector
    and is therefore suitable for recognition applications. One difference
    that this algorithm is not proportional to output. This Kohonen network
    interpretetion update only weights with non-zero output.

    Notes
    -----
    * Network architecture must contains two layers.

    Parameters
    ----------
    {full_params}

    Methods
    -------
    {unsupervised_train_epochs}
    {full_methods}

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms, layers
    >>>
    >>> np.random.seed(0)
    >>>
    >>> input_data = np.array([
    ...     [0.1961,  0.9806],
    ...     [-0.1961,  0.9806],
    ...     [0.9806,  0.1961],
    ...     [0.9806, -0.1961],
    ...     [-0.5812, -0.8137],
    ...     [-0.8137, -0.5812],
    ... ])
    >>>
    >>> kohonet = algorithms.Kohonen(
    ...     layers.LinearLayer(2) > layers.CompetitiveOutputLayer(3),
    ...     step=0.5,
    ...     verbose=False
    ... )
    >>> kohonet.train(input_data, epochs=100)
    >>> kohonet.predict(input_data)
    array([[ 0.,  1.,  0.],
           [ 0.,  1.,  0.],
           [ 1.,  0.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  0.,  1.]])
    """

    def update_indexes(self, layer_output):
        _, index_y = nonzero(layer_output)
        return index_y

    def train_epoch(self, input_train, target_train):
        input_train = format_data(input_train)

        weight = self.input_layer.weight
        predict = self.predict
        update_indexes = self.update_indexes

        for input_row in input_train:
            input_row = reshape(input_row, (1, input_row.size))
            layer_output = predict(input_row)

            index_y = update_indexes(layer_output)
            self.input_layer.weight[:, index_y] += self.step * (
                input_row.T - weight[:, index_y]
            )
