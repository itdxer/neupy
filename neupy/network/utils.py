import numpy as np


__all__ = ('iter_until_converge', 'shuffle', 'normilize_error_output', 'step')


def iter_until_converge(network, epsilon, max_epochs):
    # Trigger first iteration and store first error term
    yield network.epoch
    previous_error = error_delta = network.last_error()

    while error_delta > epsilon:
        network.epoch += 1
        yield network.epoch

        last_error = network.last_error()
        error_delta = abs(last_error - previous_error)
        previous_error = last_error

        if network.epoch >= max_epochs and error_delta > epsilon:
            network.logs.log(
                "TRAIN",
                "Epoch #{} stopped. Network didn't converge "
                "after {} iterations".format(network.epoch, max_epochs)
            )
            return

    if np.isnan(error_delta) or np.isinf(error_delta):
        network.logs.log("TRAIN", "Epoch #{} stopped. Network error value is "
                                  "invalid".format(network.epoch))
    else:
        network.logs.log("TRAIN", "Epoch #{} stopped. Network converged."
                                  "".format(network.epoch))


def shuffle(*arrays):
    if not arrays:
        return tuple()

    first = arrays[0]
    n_samples = first.shape[0]

    for array in arrays:
        if n_samples != array.shape[0]:
            raise ValueError("All matrices should have the same "
                             "number of rows")

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    arrays = list(arrays)
    for i, array in enumerate(arrays):
        arrays[i] = array[indices]

    return arrays


def normilize_error_output(output):
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
