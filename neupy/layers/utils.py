import numpy as np
import tensorflow as tf

from neupy import init
from neupy.utils import asfloat


__all__ = (
    'make_one_if_possible', 'iter_variables', 'format_variable',
    'count_parameters', 'extract_connection', 'find_variables',
)


def make_one_if_possible(shape):
    """
    Format layer's input or output shape.

    Parameters
    ----------
    shape : int or tuple

    Returns
    -------
    int or tuple
    """
    if isinstance(shape, (tuple, list)) and len(shape) == 1:
        return shape[0]
    return shape


def iter_variables(layers, only_trainable=True):
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
        for attrname, param in layer.variables.items():
            new_param = param not in observed_parameters

            if new_param and (param.trainable or not only_trainable):
                observed_parameters.append(param)
                yield layer, attrname, param


def find_variables(layers, only_trainable=False):
    parameters = []
    for _, _, parameter in iter_variables(layers, only_trainable):
        parameters.append(parameter)
    return parameters


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

    for _, _, parameter in iter_variables(connection):
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
    from neupy.layers.base import BaseGraph
    from neupy import layers

    if isinstance(instance, (list, tuple)):
        return layers.join(*instance)

    if isinstance(instance, BaseNetwork):
        return instance.connection

    if isinstance(instance, BaseGraph):
        return instance

    raise TypeError("Invalid input type. Input should be network, connection "
                    "or list of layers, got {}".format(type(instance)))


def create_shared_parameter(value, name, shape, trainable=True):
    """
    Creates NN parameter as Tensorfow variable.

    Parameters
    ----------
    value : array-like, Tensorfow variable, scalar or Initializer
        Default value for the parameter.

    name : str
        Shared variable name.

    shape : tuple
        Parameter's shape.

    trainable : bool
        Whether parameter trainable by backpropagation.

    Returns
    -------
    Tensorfow variable.
    """
    if shape is not None:
        shape = [v.value if isinstance(v, tf.Dimension) else v for v in shape]

    if isinstance(value, (int, float)):
        value = init.Constant(value)

    if isinstance(value, init.Initializer):
        value = value.sample(shape)

    if isinstance(value, tf.Variable):
        return value

    return tf.Variable(
        asfloat(value),
        name=name,
        dtype=tf.float32,
        trainable=trainable,
    )
