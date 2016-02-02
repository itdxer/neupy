from __future__ import division

import theano.tensor as T


__all__ = ('mse', 'binary_crossentropy', 'categorical_crossentropy')


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


def crossentropy(actual, predicted, epsilon):
    pass
