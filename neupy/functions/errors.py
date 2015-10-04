from __future__ import division
from numpy import log, sqrt, abs as np_abs, sum as np_sum

from neupy.utils import format_data
from neupy.functions import with_derivative


__all__ = ('mse', 'linear_error', 'cross_entropy_error', 'mae',
           'kullback_leibler', 'rmsle')


def _preformat_inputs(actual, predicted):
    actual = format_data(actual)
    predicted = format_data(predicted)

    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted values have different shapes. "
                         "Actual shape {}, predicted shape {}"
                         "".format(actual.shape, predicted.shape))

    return actual, predicted


def linear_error(actual, predicted):
    """ Linear error.
    """
    actual, predicted = _preformat_inputs(actual, predicted)
    return predicted - actual


def mae(actual, predicted):
    """ Mean absolute error.
    """
    actual, predicted = _preformat_inputs(actual, predicted)
    data_size = actual.shape[0]
    return np_abs(predicted - actual).sum() / data_size


def mse_deriv(actual, predicted):
    """ Mean square error derivative.
    """
    actual, predicted = _preformat_inputs(actual, predicted)
    data_size = actual.shape[0]
    return (2 / data_size) * (actual - predicted)


@with_derivative(mse_deriv)
def mse(actual, predicted):
    """ Mean square error.
    """
    actual, predicted = _preformat_inputs(actual, predicted)
    data_size = actual.shape[0]
    return (1 / data_size) * np_sum((predicted - actual) ** 2)


def cross_entropy_error_deriv(actual, predicted, espilon=1e-10):
    """ Cross entropy error derivative.
    """
    actual, predicted = _preformat_inputs(actual, predicted)
    count_of_inputs = actual.shape[0]
    return (
        actual - predicted
    ) / (
        count_of_inputs * actual * (1 - actual) + espilon
    )


@with_derivative(cross_entropy_error_deriv)
def cross_entropy_error(actual, predicted, espilon=1e-10):
    """ Cross entropy error.
    """
    actual, predicted = _preformat_inputs(actual, predicted)
    count_of_inputs = actual.shape[0]
    return -(1 / count_of_inputs) * np_sum(
        (
            predicted * log(actual + espilon) +
            (1 - predicted) * log(1 - actual + espilon)
        )
    )


def kullback_leibler_deriv(actual, predicted):
    """ Kullback-Leibler error derivative.
    """
    actual, predicted = _preformat_inputs(actual, predicted)
    count_of_inputs = actual.shape[0]
    return (actual - predicted) / (
        count_of_inputs * actual * (1 - actual)
    )


@with_derivative(kullback_leibler_deriv)
def kullback_leibler(actual, predicted):
    """ Kullback-Leibler error.
    """
    actual, predicted = _preformat_inputs(actual, predicted)
    count_of_inputs = actual.shape[0]
    return (1. / count_of_inputs) * np_sum(
        predicted * log(predicted / actual) +
        (1 - predicted) * log((1 - predicted) / (1 - actual))
    )


def rmsle_deriv(actual, predicted):
    """ Root mean squared logarithmic error derivative.
    """
    actual, predicted = _preformat_inputs(actual, predicted)
    count_of = predicted.shape[0]

    logarithm_difference_deriv = log(
        (actual + 1) / (predicted + 1)
    )
    return logarithm_difference_deriv / (
        count_of * (actual + 1) * (rmsle(
            actual, predicted
        ))
    )


@with_derivative(rmsle_deriv)
def rmsle(actual, predicted):
    """ Root mean squared logarithmic error.
    """
    actual, predicted = _preformat_inputs(actual, predicted)
    count_of = predicted.shape[0]
    square_logarithm_difference = log((actual + 1) / (predicted + 1)) ** 2
    return sqrt((1 / count_of) * np_sum(square_logarithm_difference))
