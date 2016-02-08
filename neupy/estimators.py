from functools import wraps

from neupy.network import errors
from neupy.utils import format_data


__all__ = ('mse', 'rmse', 'mae', 'msle', 'rmsle', 'binary_crossentropy',
           'categorical_crossentropy')


def override_theano_function(function):
    """ Override theano function and help evaluate output result.

    Parameters
    ----------
    function : function
        Function need to return theano variable.

    Returns
    -------
    function
    """
    @wraps(function)
    def wrapper(actual, expected, *args, **kwargs):
        actual = format_data(actual)
        expected = format_data(expected)

        output = function(actual, expected, *args, **kwargs)
        # use .item(0) to get a first array element and automaticaly
        # convert vector that contains one element to scalar
        return output.eval().item(0)
    return wrapper


mae = override_theano_function(errors.mae)
mse = override_theano_function(errors.mse)
rmse = override_theano_function(errors.rmse)
msle = override_theano_function(errors.msle)
rmsle = override_theano_function(errors.rmsle)
binary_crossentropy = override_theano_function(errors.binary_crossentropy)
categorical_crossentropy = override_theano_function(
    errors.categorical_crossentropy
)
