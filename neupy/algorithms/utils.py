from itertools import chain

import numpy as np
import theano.tensor as T


__all__ = ('StopTrainingException', 'shuffle', 'parameters2vector',
           'iter_parameter_values', 'setup_parameter_updates',
           'iter_until_converge', 'normalize_error')


class StopTrainingException(Exception):
    """
    Exception that needs to be triggered in case of
    early training interruption.
    """


def iter_parameter_values(network):
    """
    Iterate over all network parameters.

    Parameters
    ----------
    network : ConstructableNetwork instance

    Returns
    -------
    iterator
        Returns iterator that contains all weights and biases
        from the network. Parameters from the first layer will
        be at the beggining and the other will be in the same
        order as layers in the network.
    """
    parameters = [layer.parameters.values() for layer in network.layers]
    return chain(*parameters)


def parameters2vector(network):
    """
    Concatenate all network parameters in one big vector.

    Parameters
    ----------
    network : ConstructableNetwork instance

    Returns
    -------
    object
        Returns all parameters concatenated in one big vector.
    """
    params = iter_parameter_values(network)
    return T.concatenate([param.flatten() for param in params])


def setup_parameter_updates(parameters, parameter_update_vector):
    """
    Creates update rules for list of parameters from one vector.
    Function is useful in Conjugate Gradient or
    Levenberg-Marquardt optimization algorithms

    Parameters
    ----------
    parameters : list
        List of parameters.

    parameter_update_vector : Theano varible
        Vector that contains updates for all parameters.

    Returns
    -------
    list
        List of updates separeted for each parameter.
    """
    updates = []
    start_position = 0

    for parameter in parameters:
        end_position = start_position + parameter.size

        new_parameter = T.reshape(
            parameter_update_vector[start_position:end_position],
            parameter.shape
        )
        updates.append((parameter, new_parameter))

        start_position = end_position

    return updates


def iter_until_converge(network, epsilon, max_epochs):
    logs = network.logs

    # Trigger first iteration and store first error term
    yield network.last_epoch

    previous_error = error_delta = network.errors.last()
    epoch = network.last_epoch

    while error_delta > epsilon:
        epoch = epoch + 1
        network.last_epoch += 1

        yield epoch

        last_error = network.errors.last()
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
    Make a random shuffle for all arrays.

    Parameters
    ----------
    *arrays
        List of arrays that should be shuffled.

    Returns
    -------
    list
        List of arrays that contain shuffeled input data.
    """
    if not arrays:
        return tuple()

    arrays_without_none = [array for array in arrays if array is not None]

    if not arrays_without_none:
        return arrays

    first = arrays_without_none[0]
    n_samples = first.shape[0]

    for array in arrays_without_none:
        if n_samples != array.shape[0]:
            raise ValueError("Cannot shuffle matrices. All matrices should "
                             "have the same number of rows")

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    arrays = list(arrays)
    for i, array in enumerate(arrays):
        arrays[i] = array[indices] if array is not None else None

    if len(arrays) == 1:
        return arrays[0]

    return arrays


def normalize_error(output):
    """
    Normalize error output when result is non-scalar.

    Parameters
    ----------
    output : array-like
        Input can be any numpy array or matrix.

    Returns
    -------
    int, float, None
        Return sum of all absolute values.
    """
    if output is not None:
        return np.sum(output)
