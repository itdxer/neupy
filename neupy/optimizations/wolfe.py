"""
Main source code from Pylearn2 library:
https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/\
optimization/linesearch.py
"""
import tensorflow as tf

from neupy.utils import asfloat


def sequential_or(*conditions):
    """
    Use ``or`` operator between all conditions. Function is just
    a syntax sugar that make long Tensorfow logical conditions looks
    less ugly.

    Parameters
    ----------
    *conditions
        Conditions that returns ``True`` or ``False``
    """
    first_condition, other_conditions = conditions[0], conditions[1:]
    if not other_conditions:
        return first_condition
    return tf.logical_or(first_condition, sequential_or(*other_conditions))


def sequential_and(*conditions):
    """
    Use ``and`` operator between all conditions. Function is just
    a syntax sugar that make long Tensorfow logical conditions looks
    less ugly.

    Parameters
    ----------
    *conditions
        Conditions that returns ``True`` or ``False``
    """
    first_condition, other_conditions = conditions[0], conditions[1:]
    if not other_conditions:
        return first_condition
    return tf.logical_and(first_condition, sequential_and(*other_conditions))


def line_search(f, f_deriv, maxiter=20, c1=1e-4, c2=0.9):
    """
    Find ``x`` that satisfies strong Wolfe conditions.
    ``x > 0`` is assumed to be a descent direction.

    Parameters
    ----------
    f : callable f(x)
        Objective scalar function.

    f_deriv : callable f'(x)
        Objective function derivative.

    maxiter : int
        Maximum number of iterations. Defaults ``20``.

    c1 : float
        Parameter for Armijo condition rule. Defaults ``1e-4``.

    c2 : float
        Parameter for curvature condition rule. Defaults ``0.9``.

    Returns
    -------
    Variable
        Value ``x`` that satisfies strong Wolfe conditions and
        minimize function ``f``.

    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions.  See Wright and Nocedal, 'Numerical Optimization',
    1999, pg. 59-60.
    For the zoom phase it uses an algorithm by [...].
    """

    if not 0 < c1 < 1:
        raise ValueError("c1 should be a float between 0 and 1")

    if not 0 < c2 < 1:
        raise ValueError("c2 should be a float between 0 and 1")

    if c2 < c1:
        raise ValueError("c2 needs to be greater than c1")

    if maxiter <= 0:
        raise ValueError("maxiter needs to be greater than 0")

    c1, c2 = asfloat(c1), asfloat(c2)

    def search_iteration_step(condition, x_previous, x_current, y_previous,
                              y_current, y_deriv_previous, iteration, x_star):

        y_deriv_current = f_deriv(x_current)
        x_new = x_current * asfloat(2)
        y_new = f(x_new)

        condition1 = tf.logical_or(
            y_current > (y0 + c1 * x_current * y_deriv_0),
            tf.logical_and(
                y_current >= y_previous,
                tf.not_equal(iteration, 1),
            )
        )
        condition2 = tf.abs(y_deriv_current) <= -c2 * y_deriv_0
        condition3 = y_deriv_current >= 0

        x_star = tf.where(
            condition1,
            zoom(
                x_previous, x_current, y_previous,
                y_current, y_deriv_previous,
                f, f_deriv, y0, y_deriv_0, c1, c2
            ),
            tf.where(
                condition2,
                x_current,
                tf.where(
                    condition3,
                    zoom(
                        x_current, x_previous, y_current,
                        y_previous, y_deriv_current,
                        f, f_deriv, y0, y_deriv_0, c1, c2
                    ),
                    x_new,
                ),
            ),
        )
        y_deriv_previous_new = tf.where(
            condition1,
            y_deriv_previous,
            y_deriv_current
        )

        is_any_condition_satisfied = sequential_or(
            condition1, condition2, condition3)

        y_current_new = tf.where(
            is_any_condition_satisfied,
            y_current,
            y_new
        )
        continue_searching_condition = tf.logical_and(
            tf.not_equal(x_new, 0),
            tf.logical_not(is_any_condition_satisfied),
        )

        return [
            continue_searching_condition,
            x_current, x_new, y_current, y_current_new,
            y_deriv_previous_new, iteration + 1, x_star
        ]

    one = tf.constant(asfloat(1))
    zero = tf.constant(asfloat(0))

    x0, x1 = zero, one
    y0, y1 = f(x0), f(x1)
    y_deriv_0 = f_deriv(x0)

    outs = tf.while_loop(
        cond=lambda condition, *args: condition,
        body=search_iteration_step,
        loop_vars=[True, x0, x1, y0, y1, y_deriv_0, 1, zero],
        back_prop=False,
        maximum_iterations=maxiter,
    )
    return outs[-1]


def quadratic_minimizer(x_a, y_a, y_prime_a, x_b, y_b, bound_size_ratio=0.1):
    """
    Finds the minimizer for a quadratic polynomial that
    goes through the points (x_a, y_a), (x_b, y_b) with derivative
    at x_a of y_prime_a.

    Parameters
    ----------
    x_a : float or tensorflow variable
        Left point ``a`` in the ``x`` axis.
    y_a : float or tensorflow variable
        Output from function ``y`` at point ``a``.
    y_prime_a : float or tensorflow variable
        Output from function ``y'`` (``y`` derivative) at
        point ``a``.
    x_b : float or tensorflow variable
        Right point ``a`` in the ``x`` axis.
    y_b : float or tensorflow variable
        Output from function ``y`` at point ``b``.
    bound_size_ratio : float
        Value control acceptable bounds for interpolation. If value
        close to one of the points interpolation result will be ignored.
        The bigger ratio, the more likely to reject interpolation.
        Value needs to be between ``0`` and ``1``. Defaults to ``0.1``.

    Returns
    -------
    object
        Tensorfow variable that after evaluation is equal to
        point ``x`` which is minimizer for quadratic function.
    """

    if not 0 <= bound_size_ratio < 1:
        raise ValueError("Value ``bound_size_ratio`` need to be a float "
                         "between 0 and 1, got {}".format(bound_size_ratio))

    # The main formula works for the region [0, a] we need to
    # shift function to the left side and put point ``a``
    # at ``0`` position.
    x_range = x_b - x_a
    coef = (y_b - y_a - y_prime_a * x_range) / (x_range ** asfloat(2))
    minimizer = -y_prime_a / (asfloat(2) * coef) + x_a
    bound_size_ratio = asfloat(bound_size_ratio)

    return tf.where(
        sequential_or(
            # Handle bad cases
            tf.equal(x_range, 0),
            coef <= 0,

            tf.is_nan(minimizer),
            tf.greater(minimizer, x_b - bound_size_ratio * x_range),
            tf.less(minimizer, x_a + bound_size_ratio * x_range),
        ),
        x_a + asfloat(0.5) * x_range,
        # Since we shifted funciton to the left, we need to shift
        # the result to the right to make it correct for
        # the specified region. That's why we are adding ``x_a``
        # at the end.
        -y_prime_a / (asfloat(2) * coef) + x_a
    )


def cubic_minimizer(x_a, y_a, y_prime_a, x_b, y_b, x_c, y_c,
                    bound_size_ratio=0.2):
    """
    Finds the minimizer for a cubic polynomial that goes
    through the points (x_a, y_a), (x_b, y_b), and (x_c, y_c)
    with derivative at ``x_a`` of y_prime_a.

    Parameters
    ----------
    x_a : float or tensorflow variable
        First point ``a`` in the ``x`` axis.
    y_a : float or tensorflow variable
        Output from function ``y`` at point ``a``.
    y_prime_a : float or tensorflow variable
        Output from function ``y'`` (``y`` derivative) at
        point ``a``.
    x_b : float or tensorflow variable
        Second point ``b`` in the ``x`` axis.
    y_b : float or tensorflow variable
        Output from function ``y`` at point ``b``.
    x_c : float or tensorflow variable
        Third point ``c`` in the ``x`` axis.
    y_c : float or tensorflow variable
        Output from function ``y`` at point ``c``.
    bound_size_ratio : float
        Value control acceptable bounds for interpolation. If
        value is close to one of the points than interpolation
        result will be ignored. The bigger the ratio, the more
        likely it's going to reject interpolation. Value needs
        to be between ``0`` and ``1``. Defaults to ``0.1``.

    Returns
    -------
    object
        Tensorfow variable that after evaluation is equal to
        the point ``x`` which is a minimizer for the cubic function.
    """

    if not 0 <= bound_size_ratio < 1:
        raise ValueError("The `bound_size_ratio` value should be a float "
                         "number between 0 and 1, got {}"
                         "".format(bound_size_ratio))

    bound_size_ratio = asfloat(bound_size_ratio)

    from_a2b_dist = x_b - x_a
    from_a2c_dist = x_c - x_a

    denominator = (
        (from_a2b_dist * from_a2c_dist) ** asfloat(2) *
        (from_a2b_dist - from_a2c_dist)
    )
    tau_ab = y_b - y_a - y_prime_a * from_a2b_dist
    tau_ac = y_c - y_a - y_prime_a * from_a2c_dist

    alpha = (
        from_a2c_dist ** asfloat(2) * tau_ab -
        from_a2b_dist ** asfloat(2) * tau_ac
    ) / denominator
    beta = (
        from_a2b_dist ** asfloat(3) * tau_ac -
        from_a2c_dist ** asfloat(3) * tau_ab
    ) / denominator
    radical = beta ** asfloat(2) - asfloat(3) * alpha * y_prime_a

    minimizer = x_a + (-beta + tf.sqrt(radical)) / (asfloat(3) * alpha)

    return tf.where(
        sequential_or(
            # Handle bad cases
            radical < 0,

            tf.equal(x_a, x_b),
            tf.equal(x_a, x_c),
            tf.equal(x_b, x_c),
            tf.equal(alpha, 0),

            tf.is_nan(minimizer),
            tf.greater(minimizer, x_b - bound_size_ratio * from_a2b_dist),
            tf.less(minimizer, x_a + bound_size_ratio * from_a2b_dist),
        ),
        quadratic_minimizer(x_a, y_a, y_prime_a, x_b, y_b),
        minimizer,
    )


def zoom(x_low, x_high, y_low, y_high, y_deriv_low,
         f, f_deriv, y0, y_deriv_0, c1, c2, maxiter=10):
    """
    Notes
    -----
    Part of the optimization algorithm in `scalar_search_wolfe2`.

    Parameters
    ----------
    x_low : float
        Step size

    x_high : float
        Step size

    y_low : float
        Value of f at x_low

    y_high : float
        Value of f at x_high

    y_deriv_low : float
        Value of derivative at x_low

    f : callable f(x)
        Generates computational graph

    f_deriv : callable f'(x)
        Generates computational graph

    y0 : float
        Value of f for ``x = 0``

    y_deriv_0 : float
        Value of the derivative for ``x = 0``

    c1 : float
        Parameter for Armijo condition rule.

    c2 : float
        Parameter for curvature condition rule.

    maxiter : int
        Maximum number of iterations. Defaults to ``10``.
    """

    def zoom_itertion_step(_, x_low, y_low, y_deriv_low, x_high, y_high,
                           x_recent, y_recent, x_star):

        x_new = cubic_minimizer(
            x_low, y_low, y_deriv_low,
            x_high, y_high,
            x_recent, y_recent)

        y_new = f(x_new)
        y_deriv_new = f_deriv(x_new)

        continue_searching_condition = sequential_or(
            y_new > (y0 + c1 * x_new * y_deriv_0),
            y_new >= y_low,
            tf.abs(y_deriv_new) > (-c2 * y_deriv_0),
        )

        condition1 = tf.logical_or(
            y_new > (y0 + c1 * x_new * y_deriv_0),
            y_new >= y_low
        )
        condition2 = y_deriv_new * (x_high - x_low) >= 0

        x_recent = tf.where(
            tf.logical_or(condition1, condition2), x_high, x_low)
        y_recent = tf.where(
            tf.logical_or(condition1, condition2), y_high, y_low)
        x_high = tf.where(
            condition1, x_new, tf.where(condition2, x_low, x_high))
        y_high = tf.where(
            condition1, y_new, tf.where(condition2, y_low, y_high))

        x_low = tf.where(condition1, x_low, x_new)
        y_low = tf.where(condition1, y_low, y_new)
        y_deriv_low = tf.where(condition1, y_deriv_low, y_deriv_new)

        x_star = x_new

        return [
            continue_searching_condition,
            x_low, y_low, y_deriv_low,
            x_high, y_high,
            y_recent, x_recent,
            x_star
        ]

    zero = tf.constant(asfloat(0))
    x_recent = zero
    y_recent = y0

    outs = tf.while_loop(
        cond=lambda condition, *args: condition,
        body=zoom_itertion_step,
        loop_vars=[
            True,
            x_low, y_low, y_deriv_low,
            x_high, y_high,
            x_recent, y_recent,
            zero,
        ],
        back_prop=False,
        maximum_iterations=maxiter,
    )
    return outs[-1]
