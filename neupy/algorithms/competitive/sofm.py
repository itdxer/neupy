from __future__ import division

import numpy as np
from numpy.linalg import norm

from neupy.utils import format_data
from neupy.algorithms import Kohonen
from neupy.algorithms.associative.base import BaseAssociative
from neupy.core.docs import shared_docs
from neupy.core.properties import (IntProperty, TypedListProperty,
                                   ChoiceProperty, NumberProperty)


__all__ = ('SOFM',)


def gaussian_df(data, mean=0, std=1):
    """
    Returns gaussian density for each data sample.
    Gaussian specified by the mean and standard deviation.

    Parameters
    ----------
    data : array-like

    mean : float
        Gaussian mean.

    std : float
        Gaussian standard deviation.
    """
    if std == 0:
        return np.where(data == mean, 1, 0)

    normalizer = 2 * np.pi * std ** 2
    return np.exp(-np.square(data - mean) / normalizer)


def find_neighbour_distance(neurons, center):
    """
    Returns distance from the center into different directions
    per each dimension separately.

    Parameters
    ----------
    neurons : array-like
       Array that contains grid of n-dimensional vectors.

    center : tuple
        Index of the main neuron for which function returns
        distance to neuron's neighbours.

    Returns
    -------
    list of n-dimensional vectors
    """
    if len(center) != neurons.ndim:
        raise ValueError(
            "Cannot find center, because grid of neurons has {} dimensions "
            "and center has specified coordinates for {} dimensional grid"
            "".format(neurons.ndim, len(center)))

    slices = []
    for dim_length, center_coord in zip(neurons.shape, center):
        slices.append(slice(-center_coord, dim_length - center_coord))

    return np.ogrid[slices]


@shared_docs(find_neighbour_distance)
def gaussian_neighbours(neurons, center, std=1):
    """
    Function returns multivariate gaussian around the center
    with specified standard deviation.

    Parameters
    ----------
    {find_neighbour_distance.neurons}

    {find_neighbour_distance.center}

    std : int, float
        Gaussian standard deviation. Defaults to ``1``.
    """
    distances = find_neighbour_distance(neurons, center)
    gaussian_array = sum(gaussian_df(dist, std=std) for dist in distances)
    return gaussian_array / neurons.ndim


@shared_docs(find_neighbour_distance)
def neuron_neighbours(neurons, center, radius):
    """
    Function find all neuron's neighbours around specified
    center within a certain radius.

    Parameters
    ----------
    {find_neighbour_distance.neurons}

    {find_neighbour_distance.center}

    radius : int
        Radius indetify which neurons hear the main
        one are neighbours.

    Returns
    -------
    array-like
        Return matrix with the same dimension as ``neurons``
        where center element and it neighbours positions
        filled with value ``1`` and other as a ``0`` value.

    Examples
    --------
    >>> import numpy as np
    >>> from neupy.algorithms.competitive.sofm import neuron_neighbours
    >>>
    >>> neuron_neighbours(np.zeros((3, 3)), (0, 0), 1)
    array([[ 1.,  1.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  0.]])
    >>>
    >>> neuron_neighbours(np.zeros((5, 5)), (2, 2), 2)
    array([[ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  1.,  0.,  0.]])
    """
    distances = find_neighbour_distance(neurons, center)
    mask = sum(dist ** 2 for dist in distances) <= radius ** 2
    neurons[mask] = 1
    return neurons


def neg_euclid_distance(input_data, weight):
    """
    Negative Euclidian distance between input
    data and weight.

    Parameters
    ----------
    input_data : array-like
        Input dataset.

    weight : array-like
        Neural network's weights.

    Returns
    -------
    array-like
    """
    euclid_dist = norm(input_data.T - weight, axis=0)
    return -np.reshape(euclid_dist, (1, weight.shape[1]))


def cosine_similarity(input_data, weight):
    """
    Cosine similarity between input data and weight.

    Parameters
    ----------
    input_data : array-like
        Input dataset.

    weight : array-like
        Neural network's weights.

    Returns
    -------
    array-like
    """
    norm_prod = norm(input_data) * norm(weight, axis=0)
    summated_data = np.dot(input_data, weight)
    cosine_dist = summated_data / norm_prod
    return np.reshape(cosine_dist, (1, weight.shape[1]))


def decay_function(value, epoch, reduction_rate):
    """
    Applies to the input value monothonical decay.

    Parameters
    ----------
    value : int, float

    epoch : int
        Current training iteration (epoch).

    reduction_rate : int
        The larger the value the slower decay

    Returns
    -------
    float
    """
    return value / (1 + epoch / reduction_rate)


class SOFM(Kohonen):
    """
    Self-Organizing Feature Map (SOFM or SOM).

    Parameters
    ----------
    {BaseAssociative.n_inputs}

    n_outputs : int or None
        Number of outputs. Parameter is optional in case if
        ``feature_grid`` was specified.

        .. code-block::

            if n_outputs is None:
                n_outputs = np.prod(feature_grid)

    learning_radius : int
        Parameter defines radius within which we consider all
        neurons as neighbours to the winning neuron. The bigger
        the value the more neurons will be updated after each
        iteration.

        The ``0`` values means that we don't update
        neighbour neurons.

        Defaults to ``0``.

    std : int, float
        Parameters controls learning rate for each neighbour.
        The further neigbour neuron from the winning neuron
        the smaller that learning rate for it. Learning rate
        scales based on the factors produced by the normal
        distribution with center in the place of a winning
        neuron and stanard deviation specified as a parameter.
        The learning rate for the winning neuron is always equal
        to the value specified in the ``step`` parameter and for
        neighbour neurons it's always lower.

        The bigger the value for this parameter the bigger
        learning rate for the neighbour neurons.

        Defaults to ``1``.

    features_grid : list, tuple, None
        Feature grid defines shape of the output neurons.
        The new shape should be compatible with the number
        of outputs. It means that the following condition
        should be true:

        .. code-block:: python

            np.prod(features_grid) == n_outputs

        SOFM implementation supports n-dimensional grids.
        For instance, in order to specify grid as cube instead of
        the regular rectangular shape we can set up options as
        the following:

        .. code-block:: python

            SOFM(
                ...
                features_grid=(5, 5, 5),
                ...
            )

        Defaults to ``(n_outputs, 1)``.

    distance : {{``euclid``, ``dot_product``, ``cos``}}
        Defines function that will be used to compute
        closest weight to the input sample.

        - ``dot_product``: Just a regular dot product between
          data sample and network's weights

        - ``euclid``: Euclidian distance between data sample
          and network's weights

        - ``cos``: Cosine distance between data sample and
          network's weights

        Defaults to ``euclid``.

    reduce_radius_after : int or None
        Every specified number of epochs ``learning_radius``
        parameter will be reduced by ``1``. Process continues
        until ``learning_radius`` equal to ``0``.

        The ``None`` value disables parameter reduction
        during the training.

        Defaults to ``100``.

    reduce_step_after : int or None
        Defines reduction rate at which parameter ``step`` will
        be reduced using the following formula:

        .. code-block:: python

            step = step / (1 + current_epoch / reduce_step_after)

        The ``None`` value disables parameter reduction
        during the training.

        Defaults to ``100``.

    reduce_std_after : int or None
        Defines reduction rate at which parameter ``std`` will
        be reduced using the following formula:

        .. code-block:: python

            std = std / (1 + current_epoch / reduce_std_after)

        The ``None`` value disables parameter reduction
        during the training.

        Defaults to ``100``.

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

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms, environment
    >>>
    >>> environment.reproducible()
    >>>
    >>> data = np.array([
    ...     [0.1961, 0.9806],
    ...     [-0.1961, 0.9806],
    ...     [-0.5812, -0.8137],
    ...     [-0.8137, -0.5812],
    ... ])
    >>>
    >>> sofm = algorithms.SOFM(
    ...     n_inputs=2,
    ...     n_outputs=2,
    ...     step=0.1,
    ...     learning_radius=0,
    ...     features_grid=(2, 1),
    ... )
    >>> sofm.train(data, epochs=100)
    >>> sofm.predict(data)
    array([[0, 1],
           [0, 1],
           [1, 0],
           [1, 0]])
    """
    n_outputs = IntProperty(minval=1, allow_none=True, default=None)

    features_grid = TypedListProperty(allow_none=True, default=None)
    distance = ChoiceProperty(
        default='euclid',
        choices={
            'dot_product': np.dot,
            'euclid': neg_euclid_distance,
            'cos': cosine_similarity,
        })

    learning_radius = IntProperty(default=0, minval=0)
    std = NumberProperty(minval=0, default=1)

    reduce_radius_after = IntProperty(default=100, minval=1, allow_none=True)
    reduce_std_after = IntProperty(default=100, minval=1, allow_none=True)
    reduce_step_after = IntProperty(default=100, minval=1, allow_none=True)

    def __init__(self, **options):
        super(BaseAssociative, self).__init__(**options)

        if self.n_outputs is None and self.features_grid is None:
            raise ValueError("One of the following parameters has to be "
                             "specified: n_outputs, features_grid")

        elif self.n_outputs is None:
            self.n_outputs = np.prod(self.features_grid)

        n_grid_elements = np.prod(self.features_grid)
        invalid_feature_grid = (
            self.features_grid is not None and
            n_grid_elements != self.n_outputs)

        if invalid_feature_grid:
            raise ValueError(
                "Feature grid should contain the same number of elements "
                "as in the output layer: {0}, but found: {1} (shape: {2})"
                "".format(
                    self.n_outputs,
                    n_grid_elements,
                    self.features_grid))

        if self.features_grid is None:
            self.features_grid = (self.n_outputs, 1)

        self.init_layers()

    def predict_raw(self, input_data):
        input_data = format_data(input_data)
        n_samples = input_data.shape[0]
        output = np.zeros((n_samples, self.n_outputs))

        for i, input_row in enumerate(input_data):
            output[i, :] = self.distance(
                input_row.reshape(1, -1), self.weight)

        return output

    def update_indexes(self, layer_output):
        neuron_winner = layer_output.argmax(axis=1)
        winner_neuron_coords = np.unravel_index(
            neuron_winner, self.features_grid)

        learning_radius = self.learning_radius
        step = self.step
        std = self.std

        if self.reduce_radius_after is not None:
            learning_radius -= (self.last_epoch // self.reduce_radius_after)
            learning_radius = max(0, learning_radius)

        if self.reduce_step_after is not None:
            step = decay_function(step, self.last_epoch,
                                  self.reduce_step_after)

        if self.reduce_std_after is not None:
            std = decay_function(std, self.last_epoch,
                                 self.reduce_std_after)

        output_with_neightbours = neuron_neighbours(
            neurons=np.reshape(layer_output, self.features_grid),
            center=winner_neuron_coords,
            radius=learning_radius)

        step_scaler = gaussian_neighbours(
            np.reshape(layer_output, self.features_grid),
            winner_neuron_coords,
            std=std)

        step_scaler = step_scaler.reshape(self.n_outputs)

        index_y, = np.nonzero(
            output_with_neightbours.reshape(self.n_outputs))

        return index_y, step * step_scaler[index_y]
