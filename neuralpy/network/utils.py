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


def iter_until_converge(network, epsilon):
    epoch = network.epoch
    error = epsilon + 1

    while np.any(error > epsilon):
        yield epoch
        epoch = epoch + 1
        error = network.last_error()


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
