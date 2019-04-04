from __future__ import division

from collections import namedtuple

import six
import numpy as np
from numpy.linalg import norm

from neupy import init
from neupy.utils import as_tuple, format_data
from neupy.algorithms import Kohonen
from neupy.exceptions import WeightInitializationError
from neupy.algorithms.associative.base import BaseAssociative
from neupy.core.properties import (BaseProperty, TypedListProperty,
                                   ChoiceProperty, NumberProperty,
                                   ParameterProperty, IntProperty)
from .randomized_pca import randomized_pca
from .neighbours import (find_step_scaler_on_rect_grid,
                         find_neighbours_on_rect_grid,
                         find_neighbours_on_hexagon_grid,
                         find_step_scaler_on_hexagon_grid)


__all__ = ('SOFM',)


def neg_euclid_distance(X, weight):
    """
    Negative Euclidean distance between input
    data and weight.
    """
    euclid_dist = norm(X.T - weight, axis=0)
    return -np.expand_dims(euclid_dist, axis=0)


def cosine_similarity(X, weight):
    """
    Cosine similarity between input data and weight.
    """
    norm_prod = norm(X) * norm(weight, axis=0)
    summated_data = np.dot(X, weight)
    cosine_dist = summated_data / norm_prod
    return np.reshape(cosine_dist, (1, weight.shape[1]))


def decay_function(value, epoch, reduction_rate):
    """
    Applies to the input value monophonic decay.

    Parameters
    ----------
    value : int, float

    epoch : int
        Current training iteration (epoch).

    reduction_rate : int
        The larger the value the slower decay]
    """
    return value / (1 + epoch / reduction_rate)


class SOFMWeightParameter(ChoiceProperty):
    expected_type = as_tuple(ParameterProperty.expected_type, six.string_types)

    def __set__(self, instance, value):
        if isinstance(value, six.string_types):
            return super(ChoiceProperty, self).__set__(instance, value)
        return BaseProperty.__set__(self, instance, value)

    def __get__(self, instance, owner):
        choice_key = super(ChoiceProperty, self).__get__(instance, owner)

        if isinstance(choice_key, six.string_types):
            return self.choices[choice_key]

        return choice_key


def sample_data(data, features_grid):
    """
    Samples from the data number of rows specified in
    the ``n_outputs`` argument. In case if ``n_outputs > n_samples``
    then sample will be with replacement.

    Parameters
    ----------
    data : matrix ``(n_samples, n_features)``
        Matrix where each row is a data sample.

    features_grid : tuple
        Tuple that defines shape of the feature grid.

    Returns
    -------
    matrix ``(n_features, n_outputs)``
    """
    n_outputs = np.prod(features_grid)
    n_samples, n_features = data.shape

    with_replacement = n_samples < n_outputs
    indices = np.random.choice(n_samples, n_outputs,
                               replace=with_replacement)

    return data[indices].T


def linear_initialization(data, features_grid):
    """
    Linear weight initialization base on the randomized PCA.

    Parameters
    ----------
    data : 2d array-like

    features_grid : tuple
        Tuple that defines shape of the feature grid.

    Returns
    -------
    2d array-like
        Initialized weights
    """
    cols = features_grid[1]
    n_nodes = np.prod(features_grid)

    n_pca_components = 2
    coord = np.zeros((n_nodes, n_pca_components))

    for i in range(n_nodes):
        coord[i, 0] = int(i / cols)
        coord[i, 1] = int(i % cols)

    maximum = np.max(coord, axis=0)
    coord = 2 * (coord / maximum - 0.5)

    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    data = (data - data_mean) / data_std

    eigenvectors, eigenvalues = randomized_pca(data, n_pca_components)

    norms = np.sqrt(np.einsum('ij,ij->i', eigenvectors, eigenvectors))
    eigenvectors = ((eigenvectors.T / norms) * eigenvalues).T
    weight = data_mean + coord.dot(eigenvectors) * data_std

    return weight.T


class SOFM(Kohonen):
    """
    Self-Organizing Feature Map (SOFM or SOM).

    Notes
    -----
    - Training data samples should have normalized features.

    Parameters
    ----------
    {BaseAssociative.n_inputs}

    n_outputs : int or None
        Number of outputs. Parameter is optional in case if
        ``feature_grid`` was specified.

        .. code-block:: python

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
        The further neighbour  neuron from the winning neuron
        the smaller that learning rate for it. Learning rate
        scales based on the factors produced by the normal
        distribution with center in the place of a winning
        neuron and standard deviation specified as a parameter.
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

    grid_type : {{``rect``, ``hexagon``}}
        Defines connection type in feature grid. Type defines
        which neurons we will consider as closest to the winning
        neuron during the training.

        - ``rect`` - Connections between neurons will be organized
          in hexagonal grid.

        - ``hexagon`` - Connections between neurons will be organized
          in hexagonal grid. It works only for 1d or 2d grids.

        Defaults to ``rect``.

    distance : {{``euclid``, ``dot_product``, ``cos``}}
        Defines function that will be used to compute
        closest weight to the input sample.

        - ``dot_product``: Just a regular dot product between
          data sample and network's weights

        - ``euclid``: Euclidean distance between data sample
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

    weight : array-like, Initializer or {{``init_pca``, ``sample_from_data``}}
        Neural network weights.
        Value defined manualy should have shape ``(n_inputs, n_outputs)``.

        Also, it's possible to initialized weights base on the
        training data. There are two options:

        - ``sample_from_data`` - Before starting the training will
          randomly take number of training samples equal to number
          of expected outputs.

        - ``init_pca`` - Before training starts SOFM will applies PCA
          on a covariance matrix build from the training samples.
          Weights will be generated based on the two eigenvectors
          associated with the largest eigenvalues.

        Defaults to :class:`Normal() <neupy.init.Normal>`.

    {BaseNetwork.step}

    {BaseNetwork.show_epoch}

    {BaseNetwork.shuffle_data}

    {BaseNetwork.signals}

    {Verbose.verbose}

    Methods
    -------
    init_weights(train_data)
        Initialized weights based on the input data. It works only
        for the `init_pca` and `sample_from_data` options. For other
        cases it will throw an error.

    {BaseSkeleton.predict}

    {BaseAssociative.train}

    {BaseSkeleton.fit}

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms, utils
    >>>
    >>> utils.reproducible()
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
    ...     learning_radius=0
    ... )
    >>> sofm.train(data, epochs=100)
    >>> sofm.predict(data)
    array([[0, 1],
           [0, 1],
           [1, 0],
           [1, 0]])
    """
    n_outputs = IntProperty(minval=1, allow_none=True, default=None)
    weight = SOFMWeightParameter(
        default=init.Normal(),
        choices={
            'init_pca': linear_initialization,
            'sample_from_data': sample_data,
        }
    )
    features_grid = TypedListProperty(allow_none=True, default=None)

    DistanceParameter = namedtuple('DistanceParameter', 'name func')
    distance = ChoiceProperty(
        default='euclid',
        choices={
            'dot_product': DistanceParameter(
                name='dot_product',
                func=np.dot),
            'euclid': DistanceParameter(
                name='euclid',
                func=neg_euclid_distance),
            'cos': DistanceParameter(
                name='cosine',
                func=cosine_similarity),
        })

    GridTypeMethods = namedtuple(
        'GridTypeMethods', 'name find_neighbours find_step_scaler')

    grid_type = ChoiceProperty(
        default='rect',
        choices={
            'rect': GridTypeMethods(
                name='rectangle',
                find_neighbours=find_neighbours_on_rect_grid,
                find_step_scaler=find_step_scaler_on_rect_grid),
            'hexagon': GridTypeMethods(
                name='hexagon',
                find_neighbours=find_neighbours_on_hexagon_grid,
                find_step_scaler=find_step_scaler_on_hexagon_grid)
        }
    )

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

        if len(self.features_grid) > 2 and self.grid_type.name == 'hexagon':
            raise ValueError("SOFM with hexagon grid type should have "
                             "one or two dimensional feature grid, but got "
                             "{}d instead (shape: {!r})".format(
                                len(self.features_grid),
                                self.features_grid))

        is_pca_init = (
            isinstance(options.get('weight'), six.string_types) and
            options.get('weight') == 'init_pca'
        )

        self.initialized = False
        if not callable(self.weight):
            super(Kohonen, self).init_weights()
            self.initialized = True

            if self.distance.name == 'cosine':
                self.weight /= np.linalg.norm(self.weight, axis=0)

        elif is_pca_init and self.grid_type.name != 'rectangle':
            raise WeightInitializationError(
                "Cannot apply PCA weight initialization for non-rectangular "
                "grid. Grid type: {}".format(self.grid_type.name))

    def predict_raw(self, X):
        X = format_data(X, is_feature1d=(self.n_inputs == 1))

        if X.ndim != 2:
            raise ValueError("Only 2D inputs are allowed")

        n_samples = X.shape[0]
        output = np.zeros((n_samples, self.n_outputs))

        for i, input_row in enumerate(X):
            output[i, :] = self.distance.func(
                input_row.reshape(1, -1), self.weight)

        return output

    def update_indexes(self, layer_output):
        neuron_winner = layer_output.argmax(axis=1).item(0)
        winner_neuron_coords = np.unravel_index(
            neuron_winner, self.features_grid)

        learning_radius = self.learning_radius
        step = self.step
        std = self.std

        if self.reduce_radius_after is not None:
            learning_radius -= self.last_epoch // self.reduce_radius_after
            learning_radius = max(0, learning_radius)

        if self.reduce_step_after is not None:
            step = decay_function(step, self.last_epoch,
                                  self.reduce_step_after)

        if self.reduce_std_after is not None:
            std = decay_function(std, self.last_epoch,
                                 self.reduce_std_after)

        methods = self.grid_type
        output_grid = np.reshape(layer_output, self.features_grid)

        output_with_neighbours = methods.find_neighbours(
            grid=output_grid,
            center=winner_neuron_coords,
            radius=learning_radius)

        step_scaler = methods.find_step_scaler(
            grid=output_grid,
            center=winner_neuron_coords,
            std=std)

        index_y, = np.nonzero(
            output_with_neighbours.reshape(self.n_outputs))

        step_scaler = step_scaler.reshape(self.n_outputs)
        return index_y, step * step_scaler[index_y]

    def init_weights(self, X_train):
        if self.initialized:
            raise WeightInitializationError(
                "Weights have been already initialized")

        weight_initializer = self.weight
        self.weight = weight_initializer(X_train, self.features_grid)
        self.initialized = True

        if self.distance.name == 'cosine':
            self.weight /= np.linalg.norm(self.weight, axis=0)

    def train(self, X_train, epochs=100):
        if not self.initialized:
            self.init_weights(X_train)
        super(SOFM, self).train(X_train, epochs=epochs)

    def one_training_update(self, X_train, y_train=None):
        step = self.step
        predict = self.predict
        update_indexes = self.update_indexes

        error = 0
        for input_row in X_train:
            input_row = np.reshape(input_row, (1, input_row.size))
            layer_output = predict(input_row)

            index_y, step = update_indexes(layer_output)
            distance = input_row.T - self.weight[:, index_y]
            updated_weights = (self.weight[:, index_y] + step * distance)

            if self.distance.name == 'cosine':
                updated_weights /= np.linalg.norm(updated_weights, axis=0)

            self.weight[:, index_y] = updated_weights
            error += np.abs(distance).mean()

        return error / len(X_train)
