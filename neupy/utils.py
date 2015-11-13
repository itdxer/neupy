import sys

import theano
import numpy as np


__all__ = ('format_data', 'is_row1d', 'asfloat', 'AttributeKeyDict')


def format_data(data, row1d=False, copy=False):
    """ Transform data in a standardized format.

    Notes
    -----
    It should be applied to the input data prior to use in
    learning algorithms.

    Parameters
    ----------
    data : array-like
        Data that should be formated. That could be, matrix, vector or
        Pandas DataFrame instance.
    row1d : bool
        Defaults to ``False``.
    copy : bool
        Defaults to ``False``.

    Returns
    -------
    ndarray
        The same input data but transformed to a standardized format
        for further use.
    """
    if data is None:
        return

    data = np.array(asfloat(data), copy=copy)

    # Valid number of features for one or two dimentions
    n_features = data.shape[-1]
    if 'pandas' in sys.modules:
        pandas = sys.modules['pandas']

        if isinstance(data, (pandas.Series, pandas.DataFrame)):
            data = data.values

    if data.ndim == 1:
        data_shape = (1, n_features) if row1d else (n_features, 1)
        data = data.reshape(data_shape)

    return data


def is_row1d(layer):
    if layer is None:
        return False
    return (layer.input_size != 1)


def asfloat(value):
    """ Convert variable to float type configured by theano
    floatX variable.

    Parameters
    ----------
    value : matrix, ndarray or scalar
        Value that could be converted to float type.

    Returns
    -------
    matrix, ndarray or scalar
        Output would be input value converted to float type
        configured by theano floatX variable.
    """

    if isinstance(value, (np.matrix, np.ndarray)):
        return value.astype(theano.config.floatX)

    float_x_type = np.cast[theano.config.floatX]
    return float_x_type(value)


class AttributeKeyDict(dict):
    """ Modified built-in Python ``dict`` class. That modification
    helps get and set values like attributes.

    Exampels
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
