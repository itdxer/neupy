import numpy as np
from numpy.core.umath_tests import inner1d


__all__ = ('sign2bin', 'bin2sign', 'hopfield_energy')


def sign2bin(matrix):
    """ Convert -1 values in sign binary matrix to binary values.

    Parameters
    ----------
    matrix : array-like

    Returns
    -------
    array-like
        Return they same matrix with modified values.
    """
    return np.where(matrix == 1, 1, 0)


def bin2sign(matrix):
    """ Convert zeros in banary matrix to -1 values.

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
    """ Compute Hopfield energy between input data, output data and
    neural network weights.

    Parameters
    ----------
    input_data : vector
    output_data : vector
    weight : 2d array-like

    Returns
    -------
    float
        Hopfield energy for specific data and weights.
    """
    return -0.5 * inner1d(input_data.dot(weight), output_data)
