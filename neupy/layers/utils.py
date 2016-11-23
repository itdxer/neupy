import collections
from functools import reduce


__all__ = ('preformat_layer_shape', 'dimshuffle', 'join', 'iter_parameters',
           'count_parameters')


def preformat_layer_shape(shape):
    """
    Each layer should have input and output shape
    attributes. This function formats layer's shape value to
    make it easy to read.

    Parameters
    ----------
    shape : int or tuple

    Returns
    -------
    int or tuple
    """
    if isinstance(shape, tuple) and len(shape) == 1:
        return shape[0]
    return shape


def dimshuffle(value, ndim, axes):
    """
    Shuffle dimension based on the specified number of
    dimensions and axes.

    Parameters
    ----------
    value : Theano variable
    ndim : int
    axes : tuple, list

    Returns
    -------
    Theano variable
    """
    pattern = ['x'] * ndim
    for i, axis in enumerate(axes):
        pattern[axis] = i
    return value.dimshuffle(pattern)


def join(*connections):
    """
    Connect two layers.

    Parameters
    ----------
    *connections : layers or connections

    Returns
    -------
    connection
        Layers connected in a sequence.

    Examples
    --------
    >>> from neupy import layers
    >>> conn = layers.join(
    ...     layers.Input(784),
    ...     layers.Relu(500),
    ...     layers.Relu(300),
    ...     layers.Softmax(10),
    ... )
    >>>
    >>> conn = layers.join([
    ...     layers.Input(784),
    ...     layers.Sigmoid(100),
    ...     layers.Softmax(10),
    ... ])
    """
    from neupy.layers.connections import LayerConnection

    n_layers = len(connections)
    if n_layers == 1 and isinstance(connections[0], collections.Iterable):
        connections = connections[0]

    merged_connections = reduce(LayerConnection, connections)
    return merged_connections


def iter_parameters(layers):
    """
    Iterate through layer parameters.

    Parameters
    ----------
    layers : list of layers or connection

    Yields
    ------
    tuple
        Tuple with three ariables: (layer, attribute_name, parameter)
    """
    for layer in layers:
        for attrname, parameter in layer.parameters.items():
            yield layer, attrname, parameter


def count_parameters(connection):
    """
    Count number of parameters in Neural Network.

    Parameters
    ----------
    connection : list of laters or connection

    Returns
    -------
    int
        Number of parameters.
    """
    if not isinstance(connection, collections.Iterable):
        connection = [connection]

    n_parameters = 0
    for _, _, parameter in iter_parameters(connection):
        parameter = parameter.get_value()
        n_parameters += parameter.size
    return n_parameters
