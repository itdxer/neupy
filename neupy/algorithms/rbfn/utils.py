import math

import numpy as np
from numpy.core.umath_tests import inner1d


__all__ = ('pdf_between_data',)


def pdf_between_data(train_data, input_data, std):
    """
    Compute PDF between two samples.

    Parameters
    ----------
    train_data : array
        Training dataset.

    input_data : array
        Input dataset

    std : float
        Standard deviation for Probability Density
        Function (PDF).

    Returns
    -------
    array-like
    """
    n_train_samples = train_data.shape[0]
    n_samples = input_data.shape[0]

    results = np.zeros((n_train_samples, n_samples))
    variance = std ** 2
    const = std * math.sqrt(2 * math.pi)

    for i, input_row in enumerate(input_data):
        inputs = np.tile(input_row, (n_train_samples, 1))
        class_difference = (train_data - inputs)
        total_distance = inner1d(class_difference, class_difference)
        results[:, i] = np.exp(-total_distance / variance) / const

    return results
