import numpy as np


__all__ = ('iter_until_converge', 'shuffle', 'normalize_error', 'step',
           'StopNetworkTraining')


class StopNetworkTraining(Exception):
    """ Exception that needs to be triggered in case of
    early training interruption.
    """


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
            logs.message("TRAIN", "Epoch #{} stopped. Network didn't "
                                  "converge after {} iterations"
                                  "".format(epoch, max_epochs))
            return

    if np.isnan(error_delta) or np.isinf(error_delta):
        logs.message("TRAIN", "Epoch #{} stopped. Network error value is "
                              "invalid".format(epoch))
    else:
        logs.message("TRAIN", "Epoch #{} stopped. Network converged."
                              "".format(epoch))


def shuffle(*arrays):
    """ Make a random shuffle for all arrays.

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
            raise ValueError("All matrices should have the same "
                             "number of rows")

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    arrays = list(arrays)
    for i, array in enumerate(arrays):
        arrays[i] = array[indices] if array is not None else None

    return arrays


def normalize_error(output):
    """ Normalize error output when result is non-scalar.

    Parameters
    ----------
    output : array-like
        Input can be any numpy array or matrix.

    Returns
    -------
    int, float
        Return sum of all absolute values.
    """
    return np.sum(np.abs(output))


def step(input_value):
    """ Step function.
    """
    return np.where(input_value > 0, 1, 0)
