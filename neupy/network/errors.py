from __future__ import division

import theano.tensor as T


__all__ = ('mse', 'rmse', 'mae', 'binary_crossentropy',
           'categorical_crossentropy')


def mse(actual, predicted):
    """ Mean squared error.
    """
    return T.square(predicted - actual).mean()


def rmse(actual, predicted):
    """ Root mean squared error.
    """
    return T.sqrt(mse(actual, predicted))


def mae(actual, predicted):
    """ Mean absolute error.
    """
    return T.abs(actual - predicted).mean()


def binary_crossentropy(actual, predicted, epsilon=1e-10):
    """ Binary cross-entropy error.
    """
    predicted = T.clip(predicted, epsilon, 1.0 - epsilon)
    return T.nnet.binary_crossentropy(predicted, actual).mean()


def categorical_crossentropy(actual, predicted, epsilon=1e-10):
    """ Categorical cross-entropy error.
    """
    predicted = T.clip(predicted, epsilon, 1.0 - epsilon)
    return T.nnet.categorical_crossentropy(predicted, actual).mean()
