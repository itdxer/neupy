from functools import wraps

import tensorflow as tf

from neupy.utils.misc import as_tuple


__all__ = (
    # Main tensorflow functions
    'tensorflow_session', 'tensorflow_eval',
    'initialize_uninitialized_variables', 'function',

    # Functions that help to deal with tensorflow name scope
    'class_method_name_scope', 'function_name_scope',

    # Misc utils for tensorflow
    'flatten', 'outer', 'tf_repeat', 'dimshuffle',
    'dot', 'make_single_vector', 'setup_parameter_updates',

    # Functions that help to deal with shapes
    'shape_to_tuple', 'add_batch_dim',
)


def function(inputs, outputs, updates=None, name=None):
    """
    Function simulates behaviour of the Theano's functions.

    Parameters
    ----------
    inputs : list
        List of input placeholders

    outputs : list, Tensor
        Output that has to be computed by the function

    updates : list or None
        List of the updates that has to be performed on the variables.
        The ``None`` value means that no updates will be applied at the
        end of the computation. Defaults to ``None``.

    name : str or None
        Defaults to ``None``.

    Returns
    -------
    function
    """
    if updates is None:
        updates = []

    session = tensorflow_session()
    tensorflow_updates = []

    # Ensure that all new values has been computed. Absence of these
    # checks might lead to the non-deterministic update behaviour.
    new_values = [val[1] for val in updates if isinstance(val, (list, tuple))]

    # Make sure that all outputs has been computed
    with tf.control_dependencies(as_tuple(outputs, new_values)):
        for update in updates:
            if isinstance(update, (list, tuple)):
                old_value, new_value = update
                update = old_value.assign(new_value)
            tensorflow_updates.append(update)

        # Group variables in order to avoid output for the updates
        tensorflow_updates = tf.group(*tensorflow_updates)

    @wraps(function)
    def wrapper(*input_values):
        feed_dict = dict(zip(inputs, input_values))
        result, _ = session.run(
            [outputs, tensorflow_updates],
            feed_dict=feed_dict,
        )
        return result
    return wrapper


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

    if not variables:
        return

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
            if hasattr(method, '__self__'):  # check if method bounded
                return method(*args, **kwargs)
            return method(self, *args, **kwargs)

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
    a = tf.expand_dims(a, 1)  # column vector
    b = tf.expand_dims(b, 0)  # row vector
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

        repeats = tf.convert_to_tensor(repeats)
        return tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)


def make_single_vector(parameters):
    with tf.name_scope('parameters-vector'):
        return tf.concat([flatten(param) for param in parameters], axis=0)


def setup_parameter_updates(parameters, parameter_update_vector):
    """
    Creates update rules for list of parameters from one vector.
    Function is useful in Conjugate Gradient or
    Levenberg-Marquardt optimization algorithms

    Parameters
    ----------
    parameters : list
        List of parameters.

    parameter_update_vector : Tensorfow varible
        Vector that contains updates for all parameters.

    Returns
    -------
    list
        List of updates separeted for each parameter.
    """
    updates = []
    start_position = 0

    for parameter in parameters:
        end_position = start_position + tf.size(parameter)
        new_parameter = tf.reshape(
            parameter_update_vector[start_position:end_position],
            parameter.shape
        )
        updates.append((parameter, new_parameter))
        start_position = end_position

    return updates


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
    for dim in range(ndim):
        if dim not in axes:
            value = tf.expand_dims(value, dim)
    return value


def add_batch_dim(shape):
    return tf.TensorShape([None]).concatenate(shape)


def shape_to_tuple(shape):
    if isinstance(shape, tf.TensorShape):
        if shape.ndims is not None:
            return tuple([dim.value for dim in shape.dims])
        return None

    if isinstance(shape, list):
        return [shape_to_tuple(s) for s in shape]

    if isinstance(shape, tuple):
        return tuple([shape_to_tuple(s) for s in shape])

    return shape
