import math

import numpy as np
import tensorflow as tf

from neupy.utils import asfloat


__all__ = ('fmin_golden_search',)


def interval_location(f, minstep=1e-5, maxstep=50., maxiter=1024):
    """
    Identify interval where potentialy could be optimal step.

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
    with tf.name_scope('interval-location'):
        def find_right_bound(_, prev_func_output, step):
            func_output = f(step)
            step = tf.where(
                tf.greater(prev_func_output, func_output),
                tf.minimum(2. * step, maxstep),
                step
            )
            continue_searching = tf.logical_and(
                tf.greater_equal(prev_func_output, func_output),
                step < maxstep,
            )
            return [continue_searching, func_output, step]

        _, _, step = tf.while_loop(
            cond=lambda continue_searching, *args: continue_searching,
            body=find_right_bound,
            loop_vars=[
                True,
                tf.constant(asfloat(np.inf)),
                tf.constant(asfloat(minstep)),
            ],
            back_prop=False,
            maximum_iterations=maxiter,
        )
        return step


def golden_search(f, maxstep=50, maxiter=1024, tol=1e-5):
    """
    Identify best step for function in specific direction.

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
    with tf.name_scope('golden-search'):
        golden_ratio = asfloat((math.sqrt(5) - 1) / 2)
        tol = asfloat(tol)

        def interval_reduction(a, b, c, d):
            fc = f(c)
            fd = f(d)

            is_c_smaller_than_d = tf.less(fc, fd)

            new_a = tf.where(is_c_smaller_than_d, a, c)
            new_b = tf.where(is_c_smaller_than_d, d, b)
            new_c = tf.where(
                is_c_smaller_than_d, d - golden_ratio * (d - a), d)
            new_d = tf.where(
                is_c_smaller_than_d, c, c + golden_ratio * (b - c))

            return [new_a, new_b, new_c, new_d]

        a = tf.constant(asfloat(0))
        b = maxstep
        c = b - golden_ratio * (b - a)
        d = a + golden_ratio * (b - a)

        a, b, c, d = tf.while_loop(
            cond=lambda a, b, c, d: tf.greater(tf.abs(c - d), tol),
            body=interval_reduction,
            loop_vars=[a, b, c, d],
            back_prop=False,
            maximum_iterations=maxiter,
        )
        return (a + b) / 2


def fmin_golden_search(f, minstep=1e-5, maxstep=50., maxiter=1024, tol=1e-5):
    """
    Minimize scalar function using Golden Search.

    Parameters
    ----------
    f : func
        Function that needs to be minimized. Function has
        to return the scalar.

        .. code-block:: python

            def f(x):
                return x ** 2

    minstep : float
        Minimum step value. Defaults to ``1e-5``.

    maxstep : float
        Maximum step size. Defaults to ``50``.

    maxiter : int
        Maximum number of itrations. Defaults to ``1024``.

    tol : float
        Defaults to ``1e-5``.

    Returns
    -------
    object
        Returns the Tensorfow instance that finally should produce
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
