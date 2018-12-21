from functools import wraps

import tensorflow as tf

from neupy.utils.misc import as_tuple


__all__ = (
    'class_method_name_scope', 'function_name_scope',
    'tensorflow_session', 'tensorflow_eval', 'tf_repeat',
    'initialize_uninitialized_variables', 'flatten', 'outer', 'dot'
)


def tensorflow_session():
    if hasattr(tensorflow_session, 'cache'):
        session = tensorflow_session.cache

        if not session._closed:
            return session

    config = tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
    )
    session = tf.Session(config=config)

    tensorflow_session.cache = session
    return session


def initialize_uninitialized_variables(variables=None):
    if variables is None:
        variables = tf.global_variables()

    session = tensorflow_session()
    is_not_initialized = session.run([
        tf.is_variable_initialized(var) for var in variables])

    not_initialized_vars = [
        v for (v, f) in zip(variables, is_not_initialized) if not f]

    if len(not_initialized_vars):
        session.run(tf.variables_initializer(not_initialized_vars))


def function_name_scope(function):
    """
    Decorator that wraps any function with the name score that has the
    same name as a function.
    """
    @wraps(function)
    def wrapper(*args, **kwargs):
        with tf.name_scope(function.__name__):
            return function(*args, **kwargs)
    return wrapper


def class_method_name_scope(method):
    """
    Decorator that wraps any method with the name score that has the
    same name as a method.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with tf.name_scope(self.__class__.__name__):
            return method(*args, **kwargs)

    wrapper.original_method = method
    return wrapper


def tensorflow_eval(value):
    session = tensorflow_session()
    initialize_uninitialized_variables()
    return session.run(value)


@function_name_scope
def flatten(value):
    return tf.reshape(value, [-1])


@function_name_scope
def outer(a, b):
    a = tf.expand_dims(a, 1)
    b = tf.expand_dims(b, 0)
    return tf.matmul(a, b)


@function_name_scope
def dot(a, b):
    return tf.tensordot(a, b, 1)


def tf_repeat(tensor, repeats):
    """
    Repeat elements of an tensor. The same as ``numpy.repeat``.

    Parameters
    ----------
    input : tensor
    repeats: list, tuple
        Number of repeat for each dimension, length must be the
        same as the number of dimensions in input.

    Returns
    -------
    tensor
        Has the same type as input. Has the shape
        of ``tensor.shape * repeats``.
    """
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = as_tuple(1, repeats)
        tiled_tensor = tf.tile(expanded_tensor, multiples)
        return tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
