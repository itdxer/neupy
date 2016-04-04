from __future__ import division

import theano.tensor as T
from neupy.utils import smallest_positive_number


__all__ = ('mse', 'rmse', 'mae', 'msle', 'rmsle', 'binary_crossentropy',
           'categorical_crossentropy')


def mse(expected, predicted):
    """ Mean squared error.
    """
    return T.square(predicted - expected).mean()


def rmse(expected, predicted):
    """ Root mean squared error.
    """
    return T.sqrt(mse(expected, predicted))


def mae(expected, predicted):
    """ Mean absolute error.
    """
    return T.abs_(expected - predicted).mean()


def msle(expected, predicted):
    """ Mean squared logarithmic error.
    """
    squared_log = (T.log(predicted + 1) - T.log(expected + 1)) ** 2
    return squared_log.mean()


def rmsle(expected, predicted):
    """ Root mean squared logarithmic error.
    """
    return T.sqrt(msle(expected, predicted))


def binary_crossentropy(expected, predicted):
    """ Binary cross-entropy error.
    """
    epsilon = smallest_positive_number()
    predicted = T.clip(predicted, epsilon, 1.0 - epsilon)
    return T.nnet.binary_crossentropy(predicted, expected).mean()


def categorical_crossentropy(expected, predicted):
    """ Categorical cross-entropy error.
    """
    epsilon = smallest_positive_number()
    predicted = T.clip(predicted, epsilon, 1.0 - epsilon)
    return T.nnet.categorical_crossentropy(predicted, expected).mean()
