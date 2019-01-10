import tensorflow as tf

from neupy import init
from neupy.utils import asfloat


__all__ = ('create_shared_parameter', 'extract_network')


def extract_network(instance):
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
        return instance.network

    if isinstance(instance, BaseGraph):
        return instance

    raise TypeError(
        "Invalid input type. Input should be network, connection "
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


def count_parameters(*args, **kwargs):
    pass


def find_variables(*args, **kwargs):
    pass


def iter_variables(*args, **kwargs):
    pass
