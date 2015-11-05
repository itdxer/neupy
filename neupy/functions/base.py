from numpy import sum as np_sum, abs as np_abs, where


__all__ = ('normilize_error_output', 'step')


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
    return np_sum(np_abs(output))


def step(input_value):
    """ Step function.
    """
    return where(input_value > 0, 1, 0)
