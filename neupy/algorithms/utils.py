import numpy as np
import theano.tensor as T

from neupy.layers.utils import iter_parameters


__all__ = ('shuffle', 'parameter_values', 'iter_until_converge',
           'setup_parameter_updates')


def parameter_values(connection):
    """
    List of all trainable parameters in the network.

    Parameters
    ----------
    connection : layer, connection

    Yields
    ------
    Theano shared variable
        Network's trainable parameter.
    """
    parameters = []

    for _, _, parameter in iter_parameters(connection):
        parameters.append(parameter)

    return parameters


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
    arrays = tuple(array for array in arrays if array is not None)

    if not arrays:
        return arrays

    first = arrays[0]
    n_samples = first.shape[0]

    if any(n_samples != array.shape[0] for array in arrays):
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
