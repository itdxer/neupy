import numpy as np


__all__ = ('add_bias_column', 'iter_until_converge', 'shuffle')


def add_bias_column(data):
    """ Transform input data to matrix and add as first column which
    contains all ones.

    Parameters
    ----------
    data : array-like
        Main matrix.

    Returns
    -------
    array-like
        Copied matrix with the new column at first position which contains
        all ones.
    """
    data = np.asmatrix(data)
    bias_vector = np.ones((data.shape[0], 1))
    return np.concatenate((bias_vector, np.asarray(data)), axis=1)


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
            raise ValueError("All matrices muts have the same number of rows")

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    arrays = list(arrays)
    for i, array in enumerate(arrays):
        arrays[i] = array[indices]

    return arrays
