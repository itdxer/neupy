__all__ = ('preformat_layer_shape', 'dimshuffle', 'iter_parameters',
           'count_parameters')


def preformat_layer_shape(shape):
    """
    Format layer's input or output shape.

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


def iter_parameters(layers, only_trainable=True):
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
            if parameter.trainable or not only_trainable:
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
    n_parameters = 0

    for _, _, parameter in iter_parameters(connection):
        parameter = parameter.get_value()
        n_parameters += parameter.size

    return n_parameters
