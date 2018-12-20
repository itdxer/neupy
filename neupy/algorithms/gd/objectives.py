from __future__ import division

import tensorflow as tf

from neupy.core.docs import shared_docs
from neupy.utils import asfloat, function_name_scope


__all__ = ('mse', 'rmse', 'mae', 'msle', 'rmsle', 'binary_crossentropy',
           'categorical_crossentropy', 'binary_hinge', 'categorical_hinge')


smallest_positive_number = 1e-7  # for 32-bit float nubmers


def error_function(expected, predicted):
    """
    Parameters
    ----------
    expected : array-like, tensorflow variable
    predicted : array-like, tensorflow variable

    Returns
    -------
    array-like, tensorflow variable
    """
    raise NotImplementedError


@function_name_scope
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
    return tf.reduce_mean(tf.square(predicted - expected))


@function_name_scope
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
    return tf.sqrt(mse(expected, predicted))


@function_name_scope
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
    return tf.reduce_mean(tf.abs(expected - predicted))


@function_name_scope
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
    squared_log = tf.square(tf.log(predicted + 1) - tf.log(expected + 1))
    return tf.reduce_mean(squared_log)


@function_name_scope
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
    return tf.sqrt(msle(expected, predicted))


@function_name_scope
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
    epsilon = smallest_positive_number

    predicted = tf.clip_by_value(predicted, epsilon, 1.0 - epsilon)
    crossentropy = (
        -expected * tf.log(predicted) - (1 - expected) * tf.log(1 - predicted))

    return tf.reduce_mean(crossentropy)


@function_name_scope
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
    epsilon = smallest_positive_number

    predicted = tf.clip_by_value(predicted, epsilon, 1.0 - epsilon)
    errors = tf.reduce_sum(expected * tf.log(predicted), axis=-1)

    return -tf.reduce_mean(errors)


@function_name_scope
def binary_hinge(expected, predicted, delta=1):
    """
    Computes the binary hinge loss between predictions
    and targets.

    .. math::
        hinge(t, o) = \\max(0, \\delta - t o)

    where :math:`t=expected` and :math:`o=predicted`

    Parameters
    ----------
    expected : Tensorfow tensor
        Targets in {-1, 1} such as ground truth labels.

    predicted : Tensorfow tensor
        Predictions in (-1, 1), such as hyprbolic tangent
        output of a neural network.

    delta : scalar
        The hinge loss margin. Defaults to ``1``.

    Returns
    -------
    Tensorfow tensor
        An expression for the average binary hinge loss.

    Notes
    -----
    This is an alternative to the binary cross-entropy
    loss for binary classification problems.
    """
    error = tf.nn.relu(delta - predicted * expected)
    return tf.reduce_mean(error)


@function_name_scope
def categorical_hinge(expected, predicted, delta=1):
    """
    Computes the multi-class hinge loss between
    predictions and targets.

    .. math::
        hinge_{i}(t, o) = \\max_{j \\not = o_i} (0, t_j - t_{o_i} + \\delta)

    Parameters
    ----------
    expected : Tensorfow 2D tensor or 1D tensor
        Either a vector of int giving the correct class index
        per data point or a 2D tensor of one-hot encoding of
        the correct class in the same layout as predictions
        (non-binary targets in [0, 1] do not work!).

    predicted : Tensorfow 2D tensor
        Predictions in (0, 1), such as softmax output of
        a neural network, with data points in rows and class
        probabilities in columns.

    delta : scalar
        The hinge loss margin. Defaults to ``1``.

    Returns
    -------
    Tensorfow 1D tensor
        An expression for the average multi-class hinge loss.

    Notes
    -----
    This is an alternative to the categorical cross-entropy
    loss for multi-class classification problems.
    """
    shape = tf.shape(expected)
    n_samples = asfloat(shape[0])

    positive = tf.reduce_sum(expected * predicted, axis=-1)
    negative = tf.reduce_max((asfloat(1) - expected) * predicted, axis=-1)
    errors = tf.nn.relu(negative - positive + asfloat(1))
    return tf.reduce_sum(errors) / n_samples
