from functools import partial

from numpy import sum as np_sum, abs as np_abs


__all__ = ('with_derivative', 'deriv_partial', 'get_partial_for_func',
           'normilize_error_output', 'has_deriv')


def with_derivative(deriv_func):
    """ Decorator which add derivative property to function.

    Examples
    --------
    >>> from neupy.functions import with_derivative
    >>>
    >>> def polynomial_deriv(x):
    ...     return 2 * x + 1
    ...
    >>> @with_derivative(polynomial_deriv)
    ... def polynomial(x):
    ...     return x**2 + x + 1
    ...
    >>> polynomial(10)
    111
    >>> polynomial.deriv(10)
    21
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.deriv = deriv_func
        wrapper.func_desc = "'{}' with derivative function '{}'".format(
            func.__name__, deriv_func.__name__
        )

        return wrapper
    return decorator


def deriv_partial(func, *args, **kwargs):
    """ Override around `partial` function from `functools` library.
    This function will add the same partial arguments also for inner
    derivatives. Check all derivatives recursivly.

    Parameters
    ----------
    func : function
        Function which must override.
    *args
        list of arguments for function from parameter `func`.
    **kwargs
        dictionary of arguments for function from parameter `func`.

    Returns
    -------
    function
        The same function around `partial` function from default
        python library `functools`.
    """
    func_partial = get_partial_for_func(func.deriv)
    new_func = partial(func, *args, **kwargs)
    new_func.deriv = func_partial(func.deriv, *args, **kwargs)
    return new_func


def get_partial_for_func(function):
    """ Returns valid partial function.

    Parameters
    ----------
    function : function
        Function for which you need partial function.

    Returns
    -------
    function
        Return `partial` function from default python library `functools`
        if function hasn't derivative, and modified version otherwise.
    """
    return deriv_partial if has_deriv(function) else partial


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


def has_deriv(function):
    """ Check that function has derivative.

    Parameters
    ----------
    function : function
        Function which need to validate.

    Returns
    -------
    bool
        True if function has derivative, False otherwise.
    """
    return hasattr(function, 'deriv')
