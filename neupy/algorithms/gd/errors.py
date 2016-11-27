from __future__ import division

import theano
import theano.tensor as T

from neupy.core.docs import shared_docs
from neupy.utils import asint


__all__ = ('mse', 'rmse', 'mae', 'msle', 'rmsle', 'binary_crossentropy',
           'categorical_crossentropy', 'binary_hinge', 'categorical_hinge')


def smallest_positive_number():
    """
    Function returns different nubmer for different
    ``theano.config.floatX`` values.

    * ``1e-7`` for 32-bit float
    * ``1e-16`` for 64-bit float

    Returns
    -------
    float
        Smallest positive float number.
    """
    float_type = theano.config.floatX
    epsilon_values = {
        'float32': 1e-7,
        'float64': 1e-16,
    }
    return epsilon_values[float_type]


def error_function(expected, predicted):
    """
    Parameters
    ----------
    expected : array-like, theano variable
    predicted : array-like, theano variable

    Returns
    -------
    array-like, theano variable
    """
    raise NotImplementedError


@shared_docs(error_function)
def mse(expected, predicted):
    """
    Mean squared error.

    .. math::
        mse(t, o) = mean((t - o) ^ 2)

    where :math:`t=expected` and :math:`o=predicted`

    Parameters
    ----------
    {error_function.expected}
    {error_function.predicted}

    Returns
    -------
    {error_function.Returns}
    """
    return T.square(predicted - expected).mean()


@shared_docs(error_function)
def rmse(expected, predicted):
    """
    Root mean squared error.

    .. math::
        rmse(t, o) = \\sqrt{{mean((t - o) ^ 2)}} = \\sqrt{{mse(t, 0)}}

    where :math:`t=expected` and :math:`o=predicted`

    Parameters
    ----------
    {error_function.expected}
    {error_function.predicted}

    Returns
    -------
    {error_function.Returns}
    """
    return T.sqrt(mse(expected, predicted))


@shared_docs(error_function)
def mae(expected, predicted):
    """
    Mean absolute error.

    .. math::
        mae(t, o) = mean(\\left| t - o \\right|)

    where :math:`t=expected` and :math:`o=predicted`

    Parameters
    ----------
    {error_function.expected}
    {error_function.predicted}

    Returns
    -------
    {error_function.Returns}
    """
    return T.abs_(expected - predicted).mean()


@shared_docs(error_function)
def msle(expected, predicted):
    """
    Mean squared logarithmic error.

    .. math::
        msle(t, o) = mean((\\log(t + 1) - \\log(o + 1)) ^ 2)

    where :math:`t=expected` and :math:`o=predicted`

    Parameters
    ----------
    {error_function.expected}
    {error_function.predicted}

    Returns
    -------
    {error_function.Returns}
    """
    squared_log = T.square(T.log(predicted + 1) - T.log(expected + 1))
    return squared_log.mean()


@shared_docs(error_function)
def rmsle(expected, predicted):
    """
    Root mean squared logarithmic error.

    .. math::
        rmsle(t, o) = \\sqrt{{
            mean((\\log(t + 1) - \\log(o + 1)) ^ 2)
        }} = \\sqrt{{msle(t, o)}}

    where :math:`t=expected` and :math:`o=predicted`

    Parameters
    ----------
    {error_function.expected}
    {error_function.predicted}

    Returns
    -------
    {error_function.Returns}
    """
    return T.sqrt(msle(expected, predicted))


@shared_docs(error_function)
def binary_crossentropy(expected, predicted):
    """
    Binary cross-entropy error.

    .. math::
        crossentropy(t, o) = -(t\\cdot log(o) + (1 - t) \\cdot log(1 - o))

    where :math:`t=expected` and :math:`o=predicted`

    Parameters
    ----------
    {error_function.expected}
    {error_function.predicted}

    Returns
    -------
    {error_function.Returns}
    """
    epsilon = smallest_positive_number()
    predicted = T.clip(predicted, epsilon, 1.0 - epsilon)
    return T.nnet.binary_crossentropy(predicted, expected).mean()


@shared_docs(error_function)
def categorical_crossentropy(expected, predicted):
    """
    Categorical cross-entropy error.

    Parameters
    ----------
    {error_function.expected}
    {error_function.predicted}

    Returns
    -------
    {error_function.Returns}
    """
    epsilon = smallest_positive_number()
    predicted = T.clip(predicted, epsilon, 1.0 - epsilon)
    return T.nnet.categorical_crossentropy(predicted, expected).mean()


def binary_hinge(expected, predicted, delta=1):
    """
    Computes the binary hinge loss between predictions
    and targets.

    .. math::
        hinge(t, o) = \\max(0, \\delta - t o)

    where :math:`t=expected` and :math:`o=predicted`

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
        An expression for the average binary hinge loss.

    Notes
    -----
    This is an alternative to the binary cross-entropy
    loss for binary classification problems.
    """
    error = T.nnet.relu(delta - predicted * expected)
    return error.mean()


def categorical_hinge(expected, predicted, delta=1):
    """
    Computes the multi-class hinge loss between
    predictions and targets.

    .. math::
        hinge_{i}(t, o) = \\max_{j \\not = o_i} (0, t_j - t_{o_i} + \\delta)

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
        An expression for the average multi-class hinge loss.

    Notes
    -----
    This is an alternative to the categorical cross-entropy
    loss for multi-class classification problems.
    """
    n_classes = predicted.shape[1]

    if expected.ndim == (predicted.ndim - 1):
        expected = T.extra_ops.to_one_hot(asint(expected), n_classes)

    if expected.ndim != predicted.ndim:
        raise TypeError('Rank mismatch between expected and prediced values')

    invalid_class_indeces = expected.nonzero()
    valid_class_indeces = (1 - expected).nonzero()

    new_shape = (-1, n_classes - 1)
    rest = T.reshape(predicted[valid_class_indeces], new_shape)
    rest = T.max(rest, axis=1)

    corrects = predicted[invalid_class_indeces]
    error = T.nnet.relu(rest - corrects + delta)

    return error.mean()
