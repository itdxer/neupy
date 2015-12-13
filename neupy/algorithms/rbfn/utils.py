from math import sqrt, pi

from numpy import zeros, tile, exp
from numpy.core.umath_tests import inner1d


__all__ = ('pdf_between_data',)


def pdf_between_data(train_data, input_data, std):
    """ Compute PDF between two samples.

    Parameters
    ----------
    train_data : array
        train sample
    input_data : array
        input sample
    std : float
        standard deviation for PDF

    Returns
    -------
    array-like
    """
    # Note: This implementation works faster than 3D arrays
    # and use less memory.
    results = zeros((train_data.shape[0], input_data.shape[0]))
    variance = std ** 2
    function_const = std * sqrt(2 * pi)
    train_data_size = train_data.shape[0]

    for i, input_row in enumerate(input_data):
        inputs = tile(input_row, (train_data_size, 1))
        class_difference = (train_data - inputs)
        total_distance = inner1d(class_difference, class_difference)
        results[:, i] = exp(-total_distance / variance) / function_const

    return results
