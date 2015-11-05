from __future__ import division

import theano.tensor as T


__all__ = ('mse', 'binary_crossentropy', 'categorical_crossentropy',
           'linear_error')


def linear_error(actual, predicted):
    """ Linear error.
    """
    return predicted - actual


def mae(actual, predicted):
    """ Mean absolute error.
    """
    return T.abs(predicted - actual).mean()


def mse(actual, predicted):
    """ Mean squared error.
    """
    return T.sqr(predicted - actual).mean()


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


def kullback_leibler(actual, predicted):
    pass


def rmsle(actual, predicted):
    pass
