# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf

from neupy.utils import flatten


__all__ = ('shuffle', 'iter_until_converge', 'setup_parameter_updates',
           'make_single_vector', 'format_time')


def setup_parameter_updates(parameters, parameter_update_vector):
    """
    Creates update rules for list of parameters from one vector.
    Function is useful in Conjugate Gradient or
    Levenberg-Marquardt optimization algorithms

    Parameters
    ----------
    parameters : list
        List of parameters.

    parameter_update_vector : Tensorfow varible
        Vector that contains updates for all parameters.

    Returns
    -------
    list
        List of updates separeted for each parameter.
    """
    updates = []
    start_position = 0

    for parameter in parameters:
        end_position = start_position + tf.size(parameter)

        new_parameter = tf.reshape(
            parameter_update_vector[start_position:end_position],
            parameter.shape
        )
        updates.append((parameter, new_parameter))

        start_position = end_position

    return updates


def iter_until_converge(network, epsilon, max_epochs):
    """
    Train network until error converged or maximum number of
    epochs has been reached.

    Parameters
    ----------
    network : BaseNetwork instance

    epsilon : float
        Interrupt training in case if different absolute
        between two previous errors is less than specified
        epsilon value.

    max_epochs : int
        Maximum number of epochs to train.
    """
    logs = network.logs

    # Trigger first iteration and store first error term
    yield network.last_epoch

    previous_error = error_delta = network.training_errors[-1]
    epoch = network.last_epoch

    while error_delta > epsilon:
        epoch = epoch + 1
        network.last_epoch += 1

        yield epoch

        last_error = network.training_errors[-1]
        error_delta = abs(last_error - previous_error)
        previous_error = last_error

        if epoch >= max_epochs and error_delta > epsilon:
            logs.message("TRAIN", "Epoch #{} interrupted. Network didn't "
                                  "converge after {} iterations"
                                  "".format(epoch, max_epochs))
            return

    if np.isnan(error_delta) or np.isinf(error_delta):
        logs.message("TRAIN", "Epoch #{} interrupted. Network error value is "
                              "NaN or Inf.".format(epoch))
    else:
        logs.message("TRAIN", "Epoch #{} interrupted. Network converged."
                              "".format(epoch))


def shuffle(*arrays):
    """
    Randomly shuffle rows in the arrays qithout breaking
    associations between rows in different arrays.

    Parameters
    ----------
    *arrays
        Arrays that should be shuffled.

    Returns
    -------
    list
        List of arrays that contain shuffeled input data.
    """
    filtered_arrays = tuple(array for array in arrays if array is not None)

    if not filtered_arrays:
        return arrays

    first = filtered_arrays[0]
    n_samples = first.shape[0]

    if any(n_samples != array.shape[0] for array in filtered_arrays):
        array_shapes = [array.shape for array in filtered_arrays]
        raise ValueError("Cannot shuffle matrices. All matrices should "
                         "have the same number of rows. Input shapes are: {}"
                         "".format(array_shapes))

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    arrays = list(arrays)
    for i, array in enumerate(arrays):
        if array is not None:
            arrays[i] = array[indices]

    if len(arrays) == 1:
        return arrays[0]

    return tuple(arrays)


def make_single_vector(parameters):
    with tf.name_scope('parameters-vector'):
        return tf.concat([flatten(param) for param in parameters], axis=0)


def format_time(time):
    """
    Format seconds into human readable format.

    Parameters
    ----------
    time : float
        Time specified in seconds

    Returns
    -------
    str
        Formated time.
    """
    mins, seconds = divmod(int(time), 60)
    hours, minutes = divmod(mins, 60)

    if hours > 0:
        return '{:0>2d}:{:0>2d}:{:0>2d}'.format(hours, minutes, seconds)

    elif minutes > 0:
        return '{:0>2d}:{:0>2d}'.format(minutes, seconds)

    elif seconds > 0:
        return '{:.0f} sec'.format(seconds)

    elif time >= 1e-3:
        return '{:.0f} ms'.format(time * 1e3)

    elif time >= 1e-6:
        # microseconds
        return '{:.0f} μs'.format(time * 1e6)

    # nanoseconds or smaller
    return '{:.0f} ns'.format(time * 1e9)
