import inspect

import theano
import theano.tensor as T
from theano.tensor.var import TensorVariable
from theano.tensor.sharedvar import TensorSharedVariable
import numpy as np
from scipy.sparse import issparse


__all__ = ('format_data', 'asfloat', 'AttributeKeyDict', 'preformat_value',
           'as_tuple', 'asint', 'number_type', 'theano_random_stream')


number_type = (int, float, np.floating, np.integer)


def format_data(data, is_feature1d=True, copy=False):
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

    Returns
    -------
    ndarray
        The same input data but transformed to a standardized format
        for further use.
    """
    if data is None or issparse(data):
        return data

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
    Convert variable to float type configured by theano
    floatX variable.

    Parameters
    ----------
    value : matrix, ndarray, Theano variable or scalar
        Value that could be converted to float type.

    Returns
    -------
    matrix, ndarray, Theano variable or scalar
        Output would be input value converted to float type
        configured by theano floatX variable.
    """
    float_type = theano.config.floatX

    if isinstance(value, (np.matrix, np.ndarray)):
        if value.dtype != np.dtype(float_type):
            return value.astype(float_type)
        else:
            return value

    elif isinstance(value, (TensorVariable, TensorSharedVariable)):
        return T.cast(value, float_type)

    elif issparse(value):
        return value

    float_x_type = np.cast[float_type]
    return float_x_type(value)


def asint(value):
    """
    Convert variable to an integer type. Number of bits per
    integer depend on floatX Theano variable.

    Parameters
    ----------
    value : matrix, ndarray, Theano variable or scalar
        Value that could be converted to the integer type.

    Returns
    -------
    matrix, ndarray, Theano variable or scalar
        Output would be input value converted to the integer type.
    """
    int2float_types = {
        'float16': 'int16',
        'float32': 'int32',
        'float64': 'int64',
    }

    float_type = theano.config.floatX
    int_type = int2float_types[float_type]

    if isinstance(value, (np.matrix, np.ndarray)):
        if value.dtype != np.dtype(int_type):
            return value.astype(int_type)
        else:
            return value

    elif isinstance(value, (TensorVariable, TensorSharedVariable)):
        return T.cast(value, int_type)

    elif issparse(value):
        return value

    int_x_type = np.cast[int_type]
    return int_x_type(value)


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
        if isinstance(value, tuple):
            cleaned_values.extend(value)
        else:
            cleaned_values.append(value)
    return tuple(cleaned_values)


def theano_random_stream():
    """
    Create Theano random stream instance.
    """
    # Use NumPy seed to make Theano code easely reproducible
    max_possible_seed = 2147483647  # max 32-bit integer
    seed = np.random.randint(max_possible_seed)
    theano_random = T.shared_randomstreams.RandomStreams(seed)
    return theano_random
