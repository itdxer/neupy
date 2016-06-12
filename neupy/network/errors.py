from __future__ import division

import theano.tensor as T
from neupy.utils import smallest_positive_number


__all__ = ('mse', 'rmse', 'mae', 'msle', 'rmsle', 'binary_crossentropy',
           'categorical_crossentropy', 'binary_hinge', 'categorical_hinge')


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


def binary_hinge(expected, predicted, delta=1):
    """ Computes the binary hinge loss between predictions
    and targets.

    .. math:: L_i = \\max(0, \\delta - t_i p_i)

    Parameters
    ----------
    expected : Theano tensor
        Targets in {-1, 1} such as ground truth labels.
    predicted : Theano tensor
        Predictions in (-1, 1), such as hyprbolic tangent
        output of a neural network.
    delta : scalar
        The hinge loss margin. Defaults to ``1``.

    Returns
    -------
    Theano tensor
        An expression for the element-wise binary hinge loss.

    Notes
    -----
    This is an alternative to the binary cross-entropy
    loss for binary classification problems.
    """
    error = T.nnet.relu(delta - predicted * expected)
    return error.mean()


def categorical_hinge(expected, predicted, delta=1):
    """ Computes the multi-class hinge loss between
    predictions and targets.

    .. math:: L_i = \\max_{j \\not = p_i} (0, t_j - t_{p_i} + \\delta)

    Parameters
    ----------
    expected : Theano 2D tensor or 1D tensor
        Either a vector of int giving the correct class index
        per data point or a 2D tensor of one-hot encoding of
        the correct class in the same layout as predictions
        (non-binary targets in [0, 1] do not work!).
    predicted : Theano 2D tensor
        Predictions in (0, 1), such as softmax output of
        a neural network, with data points in rows and class
        probabilities in columns.
    delta : scalar
        The hinge loss margin. Defaults to ``1``.

    Returns
    -------
    Theano 1D tensor
        An expression for the item-wise multi-class hinge loss.

    Notes
    -----
    This is an alternative to the categorical cross-entropy
    loss for multi-class classification problems.
    """
    num_cls = predicted.shape[1]

    if expected.ndim == (predicted.ndim - 1):
        expected = T.extra_ops.to_one_hot(expected, num_cls)

    elif expected.ndim != predicted.ndim:
        raise TypeError('Rank mismatch between targets and predictions')

    invalid_class_indeces = expected.nonzero()
    valid_class_indeces = (1 - expected).nonzero()

    new_shape = (-1, num_cls - 1)
    rest = T.reshape(predicted[valid_class_indeces], new_shape)
    rest = T.max(rest, axis=1)

    corrects = predicted[invalid_class_indeces]
    error = T.nnet.relu(rest - corrects + delta)

    return error.mean()
