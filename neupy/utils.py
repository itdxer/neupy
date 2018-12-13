import inspect
from functools import wraps

import numpy as np
from scipy.sparse import issparse
import tensorflow as tf


__all__ = ('format_data', 'asfloat', 'AttributeKeyDict', 'preformat_value',
           'as_tuple', 'number_type', 'all_equal', 'class_method_name_scope',
           'tensorflow_session', 'tensorflow_eval', 'tf_repeat',
           'initialize_uninitialized_variables', 'function_name_scope')


number_type = (int, float, np.floating, np.integer)


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


def format_data(data, is_feature1d=True, copy=False, make_float=True):
    """
    Transform data in a standardized format.

    Notes
    -----
    It should be applied to the input data prior to use in
    learning algorithms.

    Parameters
    ----------
    data : array-like
        Data that should be formated. That could be, matrix, vector or
        Pandas DataFrame instance.

    is_feature1d : bool
        Should be equal to ``True`` if input data is a vector that
        contains N samples with 1 feature each. Defaults to ``True``.

    copy : bool
        Defaults to ``False``.

    make_float : bool
        If `True` then input will be converted to float.
        Defaults to ``False``.

    Returns
    -------
    ndarray
        The same input data but transformed to a standardized format
        for further use.
    """
    if data is None or issparse(data):
        return data

    if make_float:
        data = asfloat(data)

    if not isinstance(data, np.ndarray) or copy:
        data = np.array(data, copy=copy)

    # Valid number of features for one or two dimensions
    n_features = data.shape[-1]

    if data.ndim == 1:
        data_shape = (n_features, 1) if is_feature1d else (1, n_features)
        data = data.reshape(data_shape)

    return data


def asfloat(value):
    """
    Convert variable to 32 bit float number.

    Parameters
    ----------
    value : matrix, ndarray, Tensorfow variable or scalar
        Value that could be converted to float type.

    Returns
    -------
    matrix, ndarray, Tensorfow variable or scalar
        Output would be input value converted to 32 bit float.
    """
    float_type = 'float32'

    if isinstance(value, (np.matrix, np.ndarray)):
        if value.dtype != np.dtype(float_type):
            return value.astype(float_type)

        return value

    elif isinstance(value, (tf.Tensor, tf.SparseTensor)):
        return tf.cast(value, tf.float32)

    elif issparse(value):
        return value

    float_x_type = np.cast[float_type]
    return float_x_type(value)


class AttributeKeyDict(dict):
    """
    Modified built-in Python ``dict`` class. That modification
    helps get and set values like attributes.

    Examples
    --------
    >>> attrdict = AttributeKeyDict()
    >>> attrdict
    {}
    >>> attrdict.test_key = 'test_value'
    >>> attrdict
    {'test_key': 'test_value'}
    >>> attrdict.test_key
    'test_value'
    """

    def __getattr__(self, attrname):
        return self[attrname]

    def __setattr__(self, attrname, value):
        self[attrname] = value

    def __delattr__(self, attrname):
        del self[attrname]

    def __reduce__(self):
        return (self.__class__, (dict(self),))


def preformat_value(value):
    """
    Function pre-format input value depending on it's type.

    Parameters
    ----------
    value : object

    Returns
    -------
    object
    """
    if inspect.isfunction(value) or inspect.isclass(value):
        return value.__name__

    elif isinstance(value, (list, tuple, set)):
        return [preformat_value(v) for v in value]

    elif isinstance(value, (np.ndarray, np.matrix)):
        return value.shape

    elif hasattr(value, 'default'):
        return value.default

    return value


def as_tuple(*values):
    """
    Convert sequence of values in one big tuple.

    Parameters
    ----------
    *values
        Values that needs to be combined in one big tuple.

    Returns
    -------
    tuple
        All input values combined in one tuple

    Examples
    --------
    >>> as_tuple(None, (1, 2, 3), None)
    (None, 1, 2, 3, None)
    >>>
    >>> as_tuple((1, 2, 3), (4, 5, 6))
    (1, 2, 3, 4, 5, 6)
    """
    cleaned_values = []
    for value in values:
        if isinstance(value, (tuple, list)):
            cleaned_values.extend(value)
        else:
            cleaned_values.append(value)
    return tuple(cleaned_values)


def all_equal(array):
    """
    Checks if all elements in the array are equal.

    Parameters
    ----------
    array : list, tuple

    Raises
    ------
    ValueError
        If input array is empty

    Returns
    -------
    bool
        `True` in case if all elements are equal and
        `False` otherwise.
    """
    if not array:
        raise ValueError("Array is empty")

    first_item = array[0]

    if any(item != first_item for item in array):
        return False

    return True


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


def get_variable_size(variable):
    size = 1
    for dimension in variable.shape:
        size *= int(dimension)
    return size


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
