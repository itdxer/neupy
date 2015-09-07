from numpy.random import randn, random
from numpy.linalg import svd

from neupy.network.connections import LayerConnection
from neupy.helpers import import_class


__all__ = ('generate_layers', 'random_orthogonal', 'random_bounded')


DEFAULT_LAYER_CLASS = "neupy.layers.SigmoidLayer"
DEFAULT_OUTPUT_LAYER_CLASS = "neupy.layers.OutputLayer"


def generate_layers(layers_sizes):
    """ Create from list of layer sizes basic linear network.

    Parameters
    ----------
    layers_sizes : list or tuple
        Ordered lsit of network connection structure.
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


def random_orthogonal(size):
    """ Build random orthogonal 2D matrix.

    Parameters
    ----------
    size : tuple
        Generated matrix shape.
    """
    if len(size) != 2:
        raise ValueError("Size must be a tuple which contains 2"
                         "integer values.")
    rand_matrix = randn(*size)
    u, _, v = svd(rand_matrix, full_matrices=False)
    ortho_base = u if size[0] > size[1] else v
    return ortho_base[:size[0], :size[1]]


def random_bounded(size, left_bound=0, right_bound=1):
    """ Generate uniform random matrix which values between bounds.
    """
    random_weight = random(size)
    return random_weight * (right_bound - left_bound) + left_bound
