import numpy as np
import tensorflow as tf
import theano.tensor as T


__all__ = ('preformat_layer_shape', 'dimshuffle', 'iter_parameters',
           'count_parameters', 'create_input_variable', 'extract_connection')


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
    value : Tensorfow variable
    ndim : int
    axes : tuple, list

    Returns
    -------
    Tensorfow variable
    """
    dimensions_to_expand = []

    for dim in range(ndim):
        if dim not in axes:
            value = tf.expand_dims(value, dim)

    return value


def iter_parameters(layers, only_trainable=True):
    """
    Iterate through layer parameters.

    Parameters
    ----------
    layers : list of layers or connection

    only_trainable : bool
        If `True` returns only trainable parameters.
        Defaults to `True`.

    Yields
    ------
    tuple
        Tuple with three ariables: (layer, attribute_name, parameter)
    """
    observed_parameters = []

    for layer in layers:
        for attrname, parameter in layer.parameters.items():
            new_parameter = parameter not in observed_parameters

            if new_parameter and (parameter.trainable or not only_trainable):
                observed_parameters.append(parameter)
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
        shape = parameter.get_shape()
        n_parameters += np.prod(shape.as_list())

    return n_parameters


def extract_connection(instance):
    """
    Extract connection from different types of object.

    Parameters
    ----------
    instance : network, connection, list or tuple

    Returns
    -------
    connection

    Raises
    ------
    ValueError
        In case if input object doesn't have connection of layers.
    """
    # Note: Import it here in order to prevent loops
    from neupy.algorithms.base import BaseNetwork
    from neupy.layers.base import BaseConnection
    from neupy import layers

    if isinstance(instance, (list, tuple)):
        return layers.join(*instance)

    if isinstance(instance, BaseNetwork):
        return instance.connection

    if isinstance(instance, BaseConnection):
        return instance

    raise TypeError("Invalid input type. Input should be network, connection "
                    "or list of layers, got {}".format(type(instance)))
