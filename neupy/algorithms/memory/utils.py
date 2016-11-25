import numpy as np
from numpy.core.umath_tests import inner1d


__all__ = ('bin2sign', 'hopfield_energy', 'step_function')


def bin2sign(matrix):
    """
    Convert zeros in banary matrix to -1 values.

    Parameters
    ----------
    matrix : array-like

    Returns
    -------
    array-like
        Return they same matrix with modified values.
    """
    return np.where(matrix == 0, -1, 1)


def hopfield_energy(weight, input_data, output_data):
    """
    Compute Hopfield energy between input data, output data and
    neural network weights.

    Parameters
    ----------
    input_data : vector
        Input dataset

    output_data : vector
        Output dataset

    weight : 2D array
        Network's weights.

    Returns
    -------
    float
        Hopfield energy for specific data and weights.
    """
    return -0.5 * inner1d(input_data.dot(weight), output_data)


def step_function(input_value):
    """
    Step function.

    Parameters
    ----------
    input_value : array-like

    Returns
    -------
    array-like
    """
    return np.where(input_value > 0, 1, 0)
