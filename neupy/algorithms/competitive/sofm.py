from operator import mul

import numpy as np
from numpy.linalg import norm

from neupy.core.properties import (IntProperty, TypedListProperty,
                                   ChoiceProperty)
from neupy.utils import format_data
from neupy.algorithms import Kohonen


__all__ = ('SOFM',)


def neuron_neighbours(neurons, center, radius):
    """ Function find all neighbours neurons by radius and coords.

    Parameters
    ----------
    neurons : arary-like
        Array element with neurons.
    center : tuple
        Index of the main neuron for which function must find neighbours.
    radius : int
        Radius indetify which neurons hear the main one are neighbours.

    Returns
    -------
    array-like
        Return matrix with the same dimention as ``neurons`` where center
        element and it neighbours positions filled with value ``1``
        and other as ``0``.

    Examples
    --------
    >>> import numpy as np
    >>> from neupy.algorithms.competitive.sofm import neuron_neighbours
    >>>
    >>> neuron_neighbours(np.zeros((3, 3)), (0, 0), 1)
    array([[ 1.,  1.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  0.]])
    >>> neuron_neighbours(np.zeros((5, 5)), (2, 2), 2)
    array([[ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  1.,  0.,  0.]])
    """
    center_x, center_y = center
    nx, ny = neurons.shape

    y, x = np.ogrid[-center_x:nx - center_x, -center_y:ny - center_y]
    mask = (x * x + y * y) <= radius ** 2
    neurons[mask] = 1

    return neurons


def dot_product(input_data, weight):
    """ Simple dot product.

    Parameters
    ----------
    input_data : array-like
    weight : array-like

    Returns
    -------
    array-like
    """
    return input_data.dot(weight)


def neg_euclid_distance(input_data, weight):
    """ Negative Euclidian distance between input
    data and weight.

    Parameters
    ----------
    input_data : array-like
    weight : array-like

    Returns
    -------
    array-like
    """
    euclid_dist = norm(input_data.T - weight, axis=0)
    return -np.reshape(euclid_dist, (1, weight.shape[1]))


def cosine_similarity(input_data, weight):
    """ Cosine similarity between input data and weight.

    Parameters
    ----------
    input_data : array-like
    weight : array-like

    Returns
    -------
    array-like
    """
    norm_prod = norm(input_data) * norm(weight, axis=0)
    summated_data = np.dot(input_data, weight)
    cosine_dist = summated_data / norm_prod
    return np.reshape(cosine_dist, (1, weight.shape[1]))


class SOFM(Kohonen):
    """ Self-Organizing Feature Map.

    Parameters
    ----------
    learning_radius : int
        Learning radius.
    features_grid : int
        Learning radius.
    transform : {{'linear', 'euclid', 'cos'}}
        Indicate transformation operation related to the input layer.
        The ``linear`` value mean that input data would be multiplied by
        weights in typical way. The ``euclid`` method will identify the
        closest weight vector to the input one. The ``cos`` made the same
        as ``euclid``, but instead of euclid distance it uses cosine
        similarity. Defaults to ``linear``.
    {BaseAssociative.n_inputs}
    {BaseAssociative.n_outputs}
    {BaseAssociative.weight}
    {BaseNetwork.step}
    {BaseNetwork.show_epoch}
    {BaseNetwork.shuffle_data}
    {BaseNetwork.epoch_end_signal}
    {BaseNetwork.train_end_signal}
    {Verbose.verbose}

    Methods
    -------
    {BaseSkeleton.predict}
    {BaseAssociative.train}
    {BaseSkeleton.fit}
    """

    learning_radius = IntProperty(default=0, minval=0)
    features_grid = TypedListProperty()
    transform = ChoiceProperty(default='linear', choices={
        'linear': dot_product,
        'euclid': neg_euclid_distance,
        'cos': cosine_similarity,
    })

    def __init__(self, **options):
        super(SOFM, self).__init__(**options)

        invalid_feature_grid = (
            self.features_grid is not None and
            mul(*self.features_grid) != self.n_outputs
        )
        if invalid_feature_grid:
            raise ValueError(
                "Feature grid should contain the same number of elements as "
                "in the output layer: {0}, but found: {1} ({2}x{3})"
                "".format(
                    self.n_outputs,
                    mul(*self.features_grid),
                    self.features_grid[0],
                    self.features_grid[1]
                )
            )

    def init_properties(self):
        super(SOFM, self).init_properties()

        if self.features_grid is None:
            self.features_grid = (self.n_outputs, 1)

    def predict_raw(self, input_data):
        input_data = format_data(input_data)
        output = np.zeros((input_data.shape[0], self.n_outputs))
        for i, input_row in enumerate(input_data):
            output[i, :] = self.transform(input_row.reshape(1, -1),
                                          self.weight)
        return output

    def update_indexes(self, layer_output):
        neuron_winner = layer_output.argmax(axis=1)
        feature_bound = self.features_grid[1]

        output_with_neightbours = neuron_neighbours(
            np.reshape(layer_output, self.features_grid),
            (neuron_winner // feature_bound, neuron_winner % feature_bound),
            self.learning_radius
        )
        index_y, _ = np.nonzero(
            np.reshape(
                output_with_neightbours,
                (self.n_outputs, 1)
            )
        )
        return index_y
