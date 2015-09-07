import operator

from numpy import concatenate, reshape, asarray


__all__ = ('matrix_list_in_one_vector', 'vector_to_list_of_matrix')


def matrix_list_in_one_vector(matrix_list):
    """ Function concatenate all matrix from list in a single vector.

    Parameters
    ----------
    matrix_list : list of array-like elements
        List of matrices.

    Returns
    -------
    array-like
        Function will return a single vector wich contains all matrix
        transformed to the vector and concatenated in the same order
        as in the list.

    Examples
    --------
    >>> import numpy as np
    >>> from neupy.algorithms.utils import *
    >>>
    >>> a = np.arange(9).reshape((3, 3))
    >>> b = np.array([[10, 0], [0, -1]])
    >>>
    >>> matrix_list_in_one_vector([a, b])
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8, 10,  0,  0, -1])
    >>> matrix_list_in_one_vector([b, a])
    array([10,  0,  0, -1,  0,  1,  2,  3,  4,  5,  6,  7,  8])
    """
    return concatenate([m.ravel() for m in matrix_list])


def vector_to_list_of_matrix(vector, matrix_sizes):
    """ Funtion split vector into few pices and convert each one to
    a matrix.

    Parameters
    ----------
    vector : arra-like
        The vector which you need to split in few matrices.
    matrix_sizes : list
        List of new matrix shapes.

    Returns
    -------
    list
        List of all matrices built form ``vector`` parameter using
        matrix shapes list.

    Examples
    --------
    >>> import numpy as np
    >>> from neupy.algorithms.utils import *
    >>>
    >>> vec = np.array([10, 0, 0, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    >>> vector_to_list_of_matrix(vec, [(2, 2), (3, 3)])
    [array([[10,  0],
           [ 0, -1]]), array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])]
    """
    start_pos = 0
    matrix = []
    # if we get matrix it be in 2-dimention space,
    # so we need flatten it
    vector = asarray(vector).ravel()

    for size in matrix_sizes:
        vector_length = operator.mul(*size)
        end_pos = start_pos + vector_length

        reshaped_matrix = reshape(vector[start_pos:end_pos], size)
        matrix.append(reshaped_matrix)

        start_pos = end_pos

    return matrix
