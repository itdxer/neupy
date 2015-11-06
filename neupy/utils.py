import sys

import theano
import numpy as np


__all__ = ('format_data', 'is_row1d', 'asfloat', 'AttributeKeyDict')


def format_data(input_data, row1d=False, copy=False):
    if input_data is None:
        return

    input_data = np.array(input_data, copy=copy)

    # Valid number of features for one or two dimentions
    n_features = input_data.shape[-1]
    if 'pandas' in sys.modules:
        pandas = sys.modules['pandas']

        if isinstance(input_data, (pandas.Series, pandas.DataFrame)):
            input_data = input_data.values

    if input_data.ndim == 1:
        data_shape = (1, n_features) if row1d else (n_features, 1)
        input_data = input_data.reshape(data_shape)

    return input_data


def is_row1d(layer):
    if layer is None:
        return False
    return (layer.input_size != 1)


def asfloat(value):
    """ Convert variable to float type configured by theano floatX variable.

    Parameters
    ----------
    value : matrix, ndarray or scalar
        Value that could be converted to float type.

    Returns
    -------
    matrix, ndarray or scalar
        Output would be input value converted to float type configured by
        theano floatX variable.
    """

    if isinstance(value, (np.matrix, np.ndarray)):
        return value.astype(theano.config.floatX)

    float_x_type = np.cast[theano.config.floatX]
    return float_x_type(value)


class AttributeKeyDict(dict):
    """ Modified built-in Python ``dict`` class. That modification helps
    get and set values in dictionary like attributes.

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
