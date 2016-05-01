import math

import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse

from neupy.utils import asfloat


__all__ = ('fmin_golden_search',)


def interval_location(f, minstep=1e-5, maxstep=50., maxiter=1024):
    """ Identify interval where potentialy could be optimal step.

    Parameters
    ----------
    f : func
    minstep : float
        Defaults to ``1e-5``.
    maxstep : float
        Defaults to ``50``.
    maxiter : int
        Defaults to ``1024``.
    tol : float
        Defaults to ``1e-5``.

    Returns
    -------
    float
        Right bound of interval where could be optimal step in
        specified direction. In case if there is no such direction
        function return ``maxstep`` instead.
    """

    def find_right_bound(prev_func_output, step, maxstep):
        func_output = f(step)
        is_output_decrease = T.gt(prev_func_output, func_output)
        step = ifelse(
            is_output_decrease,
            T.minimum(2. * step, maxstep),
            step
        )

        is_output_increse = T.lt(prev_func_output, func_output)
        stoprule = theano.scan_module.until(
            T.or_(is_output_increse, step > maxstep)
        )
        return [func_output, step], stoprule

    (_, steps), _ = theano.scan(
        find_right_bound,
        outputs_info=[T.constant(asfloat(np.inf)),
                      T.constant(asfloat(minstep))],
        non_sequences=[maxstep],
        n_steps=maxiter
    )
    return steps[-1]


def golden_search(f, maxstep=50, maxiter=1024, tol=1e-5):
    """ Identify best step for function in specific direction.

    Parameters
    ----------
    f : func
    maxstep : float
        Defaults to ``50``.
    maxiter : int
        Defaults to ``1024``.
    tol : float
        Defaults to ``1e-5``.

    Returns
    -------
    float
        Identified optimal step.
    """

    golden_ratio = asfloat((math.sqrt(5) - 1) / 2)

    def interval_reduction(a, b, c, d, tol):
        fc = f(c)
        fd = f(d)

        a, b, c, d = ifelse(
            T.lt(fc, fd),
            [a, d, d - golden_ratio * (d - a), c],
            [c, b, d, c + golden_ratio * (b - c)]
        )

        stoprule = theano.scan_module.until(
            T.lt(T.abs_(c - d), tol)
        )
        return [a, b, c, d], stoprule

    a = T.constant(asfloat(0))
    b = maxstep
    c = b - golden_ratio * (b - a)
    d = a + golden_ratio * (b - a)

    (a, b, c, d), _ = theano.scan(
        interval_reduction,
        outputs_info=[a, b, c, d],
        non_sequences=[asfloat(tol)],
        n_steps=maxiter
    )
    return (a[-1] + b[-1]) / 2


def fmin_golden_search(f, minstep=1e-5, maxstep=50., maxiter=1024, tol=1e-5):
    """ Minimize scalar function using Golden Search.

    Parameters
    ----------
    f : func
        Function that needs to be minimized. Function need to
        return the scalar.
    minstep : float
        Defaults to ``1e-5``.
    maxstep : float
        Defaults to ``50``.
    maxiter : int
        Defaults to ``1024``.
    tol : float
        Defaults to ``1e-5``.

    Returns
    -------
    object
        Returns the Theano instance that finally should produce
        best possbile step for specified function.
    """
    params = (
        ('maxiter', maxiter),
        ('minstep', minstep),
        ('maxstep', maxstep),
        ('tol', tol),
    )

    for param_name, param_value in params:
        if param_value <= 0:
            raise ValueError("Parameter `{}` should be greater than zero."
                             "".format(param_name))

    if minstep >= maxstep:
        raise ValueError("`minstep` should be smaller than `maxstep`")

    maxstep = interval_location(f, minstep, maxstep, maxiter)
    best_step = golden_search(f, maxstep, maxiter, tol)

    return best_step
