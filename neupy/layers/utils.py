from numpy.random import randn, random
from numpy.linalg import svd

from neupy.network.connections import LayerConnection
from neupy.helpers import import_class


__all__ = ('generate_layers', 'random_orthogonal', 'random_bounded',
           'generate_weight')


DEFAULT_LAYER_CLASS = "neupy.layers.Sigmoid"
DEFAULT_OUTPUT_LAYER_CLASS = "neupy.layers.Output"


def generate_layers(layers_sizes):
    """ Create from list of layer sizes basic linear network.

    Parameters
    ----------
    layers_sizes : list or tuple
        Ordered list of network connection structure.

    Returns
    -------
    LayerConnection
        Constructed connection.
    """

    if len(layers_sizes) < 2:
        raise ValueError("Network must contains at least 2 layers.")

    default_layer_class = import_class(DEFAULT_LAYER_CLASS)
    default_output_layer_class = import_class(DEFAULT_OUTPUT_LAYER_CLASS)

    output_layer_size = layers_sizes.pop()
    connection = default_output_layer_class(output_layer_size)

    for input_size in reversed(layers_sizes):
        left_layer = default_layer_class(input_size)
        connection = LayerConnection(left_layer, connection)

    return connection


def random_orthogonal(shape):
    """ Build random orthogonal 2D matrix.

    Parameters
    ----------
    shape : tuple
        Generated matrix shape.

    Returns
    -------
    ndarray
        Orthogonalized random matrix.
    """

    if len(shape) not in (1, 2):
        raise ValueError("Shape attribute should be a tuple which contains "
                         "1 or 2 integer values.")

    rand_matrix = randn(*shape)

    if len(shape) == 1:
        return rand_matrix

    nrows, ncols = shape
    u, _, v = svd(rand_matrix, full_matrices=False)
    ortho_base = u if nrows > ncols else v

    return ortho_base[:nrows, :ncols]


def random_bounded(shape, bounds=(0, 1)):
    """ Generate uniform random matrix which values between bounds.

    Parameters
    ----------
    shape : tuple
        Generated matrix shape.
    bounds : list or tuple
        Set up generated weights between bounds. Values should be tuple or
        list that contains two numbers. Defaults to ``(0, 1)``.

    Returns
    -------
    ndarray
        Random matrix between selected bounds.
    """

    if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2):
        raise ValueError("`{}` is invalid value for `bounds` parameter. "
                         "Value should be a tuple or a list that contains "
                         "two numbers.".format(bounds))

    left_bound, right_bound = bounds
    random_weight = random(shape)
    return random_weight * (right_bound - left_bound) + left_bound


GAUSSIAN = 'gauss'
BOUNDED = 'bounded'
ORTHOGONAL = 'ortho'

VALID_INIT_METHODS = (GAUSSIAN, BOUNDED, ORTHOGONAL)


def generate_weight(shape, bounds=None, init_method=GAUSSIAN):
    """ Generate random weights for neural network connections.

    Parameters
    ----------
    shape : tuple of int
        Weight shape.
    bounds : tuple of int
        Available only for ``init_method`` equal to ``bounded``.
        Value identify minimum and maximum possible value in random weights.
    init_method : {{'bounded', 'gauss', 'ortho'}}
        Weight initialization method.
        ``gauss`` will generate random weights from Standard Normal
        Distribution.
        ``bounded`` generate random weights from Uniform distribution.
        ``ortho`` generate random orthogonal matrix.
        Defaults to ``gauss``.

    Returns
    -------
    ndarray
        Random weight.
    """

    if init_method not in VALID_INIT_METHODS:
        raise ValueError("Undefined initialization method `{}`. Available: {}"
                         "".format(init_method, VALID_INIT_METHODS))

    if init_method == GAUSSIAN:
        weight = randn(*shape)

    elif init_method == BOUNDED:
        weight = random_bounded(shape, bounds)

    elif init_method == ORTHOGONAL:
        weight = random_orthogonal(shape)

    return weight
