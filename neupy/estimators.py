from functools import wraps

import numpy as np

from neupy.network import errors
from neupy.utils import format_data


__all__ = ('rmsle', 'mse', 'binary_crossentropy', 'categorical_crossentropy')


def override_theano_function(function):
    @wraps(function)
    def wrapper(actual, expected, *args, **kwargs):
        actual = format_data(actual)
        expected = format_data(expected)

        output = function(actual, expected, *args, **kwargs)
        return output.eval()
    return wrapper


def rmsle(actual, expected):
    """ Root mean square logarithmic error.

    Parameters
    ----------
    actual : array-like
    expected : array-like

    Returns
    -------
    float
        Computed error using actual and expected values.
    """

    actual = format_data(actual)
    expected = format_data(expected)

    count_of = expected.shape[0]
    square_logarithm_difference = np.log((actual + 1) / (expected + 1)) ** 2
    return np.sqrt((1. / count_of) * np.sum(square_logarithm_difference))


mse = override_theano_function(errors.mse)
binary_crossentropy = override_theano_function(errors.binary_crossentropy)
categorical_crossentropy = override_theano_function(
    errors.categorical_crossentropy
)
