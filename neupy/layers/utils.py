__all__ = ('preformat_layer_shape', 'dimshuffle')


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
