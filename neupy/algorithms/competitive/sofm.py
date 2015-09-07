from operator import mul

from numpy import ogrid, reshape, nonzero

from neupy.core.properties import NonNegativeIntProperty, NumberBoundProperty
from neupy.layers import CompetitiveOutputLayer
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

    y, x = ogrid[-center_x:nx - center_x, -center_y:ny - center_y]
    mask = (x * x + y * y) <= radius ** 2
    neurons[mask] = 1

    return neurons


class SOFM(Kohonen):
    """ Self-Organizing Feature Map.

    Notes
    -----
    * Network architecture must contains two layers.
    * Second layer must be :layer:`CompetitiveOutputLayer`.

    Parameters
    ----------
    learning_radius : int
        Learning radius.
    features_grid : int
        Learning radius.
    {full_params}

    Methods
    -------
    {unsupervised_train_epochs}
    {predict}
    {plot_errors}
    {last_error}
    """
    learning_radius = NonNegativeIntProperty(default=0)
    features_grid = NumberBoundProperty()
    # # None - mean that this property is the same as default step
    # neighbours_step = NumberProperty()

    def __init__(self, connection, **options):
        super(SOFM, self).__init__(connection, **options)

        if not isinstance(self.output_layer, CompetitiveOutputLayer):
            raise ValueError("Output layer must be `CompetitiveOutputLayer`")

        if self.features_grid is not None:
            if mul(*self.features_grid) != self.output_layer.input_size:
                raise ValueError(
                    "Feature grid must contains the same size of elements as "
                    "at output layer: {0}. But it contains: {1} "
                    "({2}x{3})".format(
                        self.output_layer.input_size,
                        mul(*self.features_grid),
                        self.features_grid[0],
                        self.features_grid[1]
                    )
                )

    def setup_defaults(self):
        super(SOFM, self).setup_defaults()

        # if self.neighbours_step is None:
        #     self.neighbours_step = self.step

        if self.features_grid is None:
            self.features_grid = (self.output_layer.input_size, 1)

    def update_indexes(self, layer_output):
        neuron_winner = layer_output.argmax(axis=1)
        feature_bound = self.features_grid[1]

        output_with_neightbours = neuron_neighbours(
            reshape(layer_output, self.features_grid),
            (neuron_winner // feature_bound, neuron_winner % feature_bound),
            self.learning_radius
        )
        index_y, _ = nonzero(
            reshape(
                output_with_neightbours,
                (self.output_layer.input_size, 1)
            )
        )
        return index_y
