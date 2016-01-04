import theano
import theano.tensor as T
from theano.ifelse import ifelse
import numpy as np

from neupy.utils import asfloat


one = asfloat(1)
zero = T.constant(asfloat(0))
nan = T.constant(asfloat(np.nan))


def sequential_or(*conditions):
    """ Use ``or`` operator between all conditions. Function is just
    a syntax sugar that make long Theano logical conditions looks
    less ugly.

    Parameters
    ----------
    *conditions
        Conditions that returns ``True`` or ``False``
    """
    first_condition, other_conditions = conditions[0], conditions[1:]
    if not other_conditions:
        return first_condition
    return T.or_(first_condition, sequential_or(*other_conditions))


def sequential_and(*conditions):
    """ Use ``and`` operator between all conditions. Function is just
    a syntax sugar that make long Theano logical conditions looks
    less ugly.

    Parameters
    ----------
    *conditions
        Conditions that returns ``True`` or ``False``
    """
    first_condition, other_conditions = conditions[0], conditions[1:]
    if not other_conditions:
        return first_condition
    return T.and_(first_condition, sequential_and(*other_conditions))


def scalar_search_wolfe2(phi, derphi, phi0=None, old_phi0=None, derphi0=None,
                         maxiter=20, c1=1e-4, c2=0.9):
    """ Find alpha that satisfies strong Wolfe conditions.
    alpha > 0 is assumed to be a descent direction.

    Parameters
    ----------
    phi : callable f(x)
        Objective scalar function.
    derphi : callable f'(x)
        Objective function derivative (can be None)
    phi0 : float, optional
        Value of phi at s=0
    old_phi0 : float, optional
        Value of phi at previous point
    derphi0 : float, optional
        Value of derphi at s=0
    c1 : float
        Parameter for Armijo condition rule.
    c2 : float
        Parameter for curvature condition rule.

    Returns
    -------
    alpha_star : float
        Best alpha
    phi_star : WRITEME
        phi at alpha_star
    phi0 : WRITEME
        phi at 0
    derphi_star : WRITEME
        derphi at alpha_star

    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions.  See Wright and Nocedal, 'Numerical Optimization',
    1999, pg. 59-60.
    For the zoom phase it uses an algorithm by [...].
    """

    if phi0 is None:
        phi0 = phi(zero)

    if derphi0 is None and derphi is not None:
        derphi0 = derphi(zero)

    alpha0 = zero

    if old_phi0 is not None:
        alpha1 = T.minimum(one, asfloat(2.02) * (phi0 - old_phi0) / derphi0)
    else:
        old_phi0 = nan
        alpha1 = one

    alpha1 = T.switch(alpha1 < zero, one, alpha1)

    # This shouldn't happen. Perhaps the increment has slipped below
    # machine precision?  For now, set the return variables skip the
    # useless while loop, and raise warnflag=2 due to possible imprecision.
    phi0 = T.switch(T.eq(alpha1, zero), old_phi0, phi0)
    phi_a1 = ifelse(T.eq(alpha1, zero), phi0, phi(alpha1))

    phi_a0 = phi0
    derphi_a0 = derphi0

    # Make sure variables are tensors otherwise strange things happen
    c1 = T.as_tensor_variable(c1)
    c2 = T.as_tensor_variable(c2)

    def while_search(alpha0, alpha1, phi_a0, phi_a1, derphi_a0, i_t,
                    alpha_star, phi_star, derphi_star):
        derphi_a1 = derphi(alpha1)
        cond1 = T.bitwise_or(phi_a1 > phi0 + c1 * alpha1 * derphi0,
                              T.bitwise_and(phi_a1 >= phi_a0, i_t > zero))
        cond2 = abs(derphi_a1) <= -c2 * derphi0
        cond3 = derphi_a1 >= zero
        alpha_star_c1, phi_star_c1, derphi_star_c1 = \
                zoom(alpha0, alpha1, phi_a0, phi_a1, derphi_a0,
                      phi, derphi, phi0, derphi0, c1, c2)
        alpha_star_c3, phi_star_c3, derphi_star_c3 = \
                zoom(alpha1, alpha0, phi_a1, phi_a0, derphi_a1, phi,
                      derphi, phi0, derphi0, c1, c2)
        nw_alpha1 = alpha1 * np.asarray(2, dtype=theano.config.floatX)
        nw_phi = phi(nw_alpha1)
        alpha_star, phi_star, derphi_star = \
                ifelse(cond1,
                          (alpha_star_c1, phi_star_c1, derphi_star_c1),
                ifelse(cond2,
                          (alpha1, phi_a1, derphi_a1),
                ifelse(cond3,
                          (alpha_star_c3, phi_star_c3, derphi_star_c3),
                           (nw_alpha1, nw_phi, nan),
                      name='alphastar_c3'),
                      name='alphastar_c2'),
                      name='alphastar_c1')

        return ([alpha1,
                 nw_alpha1,
                 phi_a1,
                 ifelse(sequential_or(cond1, cond2, cond3),
                        phi_a1,
                        nw_phi,
                        name='nwphi1'),
                 ifelse(cond1, derphi_a0, derphi_a1, name='derphi'),
                 i_t + one,
                 alpha_star,
                 phi_star,
                 derphi_star],
                 theano.scan_module.scan_utils.until(
                    sequential_or(T.eq(nw_alpha1, zero),
                                  cond1,
                                  cond2,
                                  cond3))
                 )
    states = [alpha0, alpha1, phi_a0, phi_a1, derphi_a0,
              zero, zero, zero, zero]

    outs, updates = theano.scan(while_search,
                                outputs_info=states,
                                n_steps=maxiter,
                                name='while_search')

    out3 = outs[-3][-1]
    out2 = outs[-2][-1]
    out1 = outs[-1][-1]
    alpha_star, phi_star, derphi_star = \
            ifelse(T.eq(alpha1, zero),
                        (nan, phi0, nan),
                        (out3, out2, out1), name='main_alphastar')
    return alpha_star, phi_star,  phi0, derphi_star


def quadratic_minimizer(x_a, y_a, y_prime_a, x_b, y_b):
    """ Finds the minimizer for a quadratic polynomial that
    goes through the points (x_a, y_a), (x_b, y_b) with derivative
    at x_a of y_prime_a.

    Parameters
    ----------
    x_a : float or theano variable
        Left point ``a`` in the ``x`` axis.
    y_a : float or theano variable
        Output from function ``y`` at point ``a``.
    y_prime_a : float or theano variable
        Output from function ``y'`` (``y`` derivative) at
        point ``a``.
    x_b : float or theano variable
        Right point ``a`` in the ``x`` axis.
    y_b : float or theano variable
        Output from function ``y`` at point ``b``.

    Returns
    -------
    object
        Theano variable that after evaluation is equal to
        point ``x`` which is minimizer for quadratic function.
    """

    # The main formula works for the region [0, a] we need to
    # shift function to the left side and put point ``a``
    # at ``0`` position.
    x_range = x_b - x_a
    coef = (y_b - y_a - y_prime_a * x_range) / (x_range ** 2)

    return T.switch(
        T.or_(T.eq(x_range, zero), coef <= zero),
        nan,
        # Since we shifted funciton to the left, we need to shift
        # the result to the right to make it correct for
        # the specified region. That's why we are adding ``x_a``
        # at the end.
        -y_prime_a / (asfloat(2) * coef) + x_a
    )


def cubic_minimizer(x_a, y_a, y_prime_a, x_b, y_b, x_c, y_c):
    """ Finds the minimizer for a cubic polynomial that goes
    through the points (x_a, y_a), (x_b, y_b), and (x_c, y_c) with
    derivative at ``x_a`` of y_prime_a. If no minimizer can be
    found return ``NaN``.

    Parameters
    ----------
    x_a : float or theano variable
        First point ``a`` in the ``x`` axis.
    y_a : float or theano variable
        Output from function ``y`` at point ``a``.
    y_prime_a : float or theano variable
        Output from function ``y'`` (``y`` derivative) at
        point ``a``.
    x_b : float or theano variable
        Second point ``b`` in the ``x`` axis.
    y_b : float or theano variable
        Output from function ``y`` at point ``b``.
    x_c : float or theano variable
        Third point ``c`` in the ``x`` axis.
    y_c : float or theano variable
        Output from function ``y`` at point ``c``.

    Returns
    -------
    object
        Theano variable that after evaluation is equal to
        point ``x`` which is minimizer for cubic function.
    """

    from_a2b_dist = x_b - x_a
    from_a2c_dist = x_c - x_a

    denominator = (
        (from_a2b_dist * from_a2c_dist) ** 2 *
        (from_a2b_dist - from_a2c_dist)
    )
    tau_ab = y_b - y_a - y_prime_a * from_a2b_dist
    tau_ac = y_c - y_a - y_prime_a * from_a2c_dist

    alpha = (
        from_a2c_dist ** 2 * tau_ab -
        from_a2b_dist ** 2 * tau_ac
    ) / denominator
    beta = (
        from_a2b_dist ** 3 * tau_ac -
        from_a2c_dist ** 3 * tau_ab
    ) / denominator
    radical = beta ** 2 - 3 * alpha * y_prime_a

    return ifelse(
        sequential_or(
            radical < zero,
            T.eq(x_a, x_b),
            T.eq(x_a, x_c),
            T.eq(x_b, x_c),
            T.eq(alpha, zero)
        ),
        nan,
        x_a + (-beta + T.sqrt(radical)) / (asfloat(3) * alpha)
    )


def zoom(a_low, a_high, phi_low, phi_high, derphi_low,
          phi, derphi, phi0, derphi0, c1, c2, maxiter=10):
    """
    Notes
    -----
    Part of the optimization algorithm in `scalar_search_wolfe2`.

    Parameters
    ----------
    a_low : float
        Step size
    a_high : float
        Step size
    phi_low : float
        Value of f at a_low
    phi_high : float
        Value of f at a_high
    derphi_low : float
        Value of derivative at a_low
    phi : callable
        Generates computational graph
    derphi : callable
        Generates computational graph
    phi0 : float
        Value of f at 0
    derphi0 : float
        Value of the derivative at 0
    c1 : float
        Wolfe parameter
    c2 : float
        Wolfe parameter
    """

    def zoom_itertion_step(phi_rec, a_rec, a_low, a_high, phi_high, phi_low,
                           derphi_low, a_star, val_star, valprime):
        # interpolate to find a trial step length between a_low and
        # a_high Need to choose interpolation here.  Use cubic
        # interpolation and then if the result is within delta *
        # dalpha or outside of the interval bounded by a_low or a_high
        # then use quadratic interpolation, if the result is still too
        # close, then use bisection
        dalpha = a_high - a_low
        a = T.switch(dalpha < zero, a_high, a_low)
        b = T.switch(dalpha < zero, a_low, a_high)

        # minimizer of cubic interpolant
        # (uses phi_low, derphi_low, phi_high, and the most recent value of phi)
        #
        # if the result is too close to the end points (or out of the
        # interval) then use quadratic interpolation with phi_low,
        # derphi_low and phi_high if the result is stil too close to the
        # end points (or out of the interval) then use bisection

        # cubic interpolation
        cchk = asfloat(0.2) * dalpha
        a_j_cubic = cubic_minimizer(a_low, phi_low, derphi_low,
                                    a_high, phi_high, a_rec, phi_rec)
        # quadric interpolation
        qchk = asfloat(0.1) * dalpha
        a_j_quad = quadratic_minimizer(a_low, phi_low, derphi_low, a_high, phi_high)
        cond_q = sequential_or(T.isnan(a_j_quad),
                         a_j_quad > b - qchk,
                         a_j_quad < a + qchk)
        a_j_quad = T.switch(cond_q, a_low + asfloat(0.5) * dalpha, a_j_quad)

        # pick between the two ..
        cond_c = sequential_or(T.isnan(a_j_cubic),
                               a_j_cubic > b - cchk,
                               a_j_cubic < a + cchk)
        # this lazy if actually decides if we need to run the quadric
        # interpolation
        a_j = T.switch(cond_c, a_j_quad, a_j_cubic)
        #a_j = ifelse(cond_c, a_j_quad,  a_j_cubic)

        # Check new value of a_j
        phi_aj = phi(a_j)
        derphi_aj = derphi(a_j)

        stop = sequential_and(phi_aj <= phi0 + c1 * a_j * derphi0,
                              phi_aj < phi_low,
                              abs(derphi_aj) <= -c2 * derphi0)

        cond1 = T.bitwise_or(phi_aj > phi0 + c1 * a_j * derphi0,
                              phi_aj >= phi_low)
        cond2 = derphi_aj * (a_high - a_low) >= zero

        # Switches just make more sense here because they have a C
        # implementation and they get composed
        phi_rec = ifelse(cond1, phi_high, T.switch(cond2, phi_high, phi_low))
        a_rec = ifelse(cond1, a_high, T.switch(cond2, a_high, a_low))
        a_high = ifelse(cond1, a_j, T.switch(cond2, a_low, a_high))
        phi_high = ifelse(cond1, phi_aj, T.switch(cond2, phi_low, phi_high))

        a_low = T.switch(cond1, a_low, a_j)
        phi_low = T.switch(cond1, phi_low, phi_aj)
        derphi_low = ifelse(cond1, derphi_low, derphi_aj)

        a_star = a_j
        val_star = phi_aj
        valprime = ifelse(cond1, nan, T.switch(cond2, derphi_aj, nan))

        return ([phi_rec,
                 a_rec,
                 a_low,
                 a_high,
                 phi_high,
                 phi_low,
                 derphi_low,
                 a_star,
                 val_star,
                 valprime],
                 theano.scan_module.scan_utils.until(stop))

    phi_rec = phi0
    a_rec = zero

    dalpha = a_high - a_low
    a = T.switch(dalpha < zero, a_high, a_low)
    b = T.switch(dalpha < zero, a_low, a_high)

    # minimizer of cubic interpolant
    # (uses phi_low, derphi_low, phi_high, and the most recent value of phi)
    #
    # if the result is too close to the end points (or out of the
    # interval) then use quadratic interpolation with phi_low,
    # derphi_low and phi_high if the result is stil too close to the
    # end points (or out of the interval) then use bisection

    # quadric interpolation
    qchk = asfloat(0.1) * dalpha
    a_j = quadratic_minimizer(a_low, phi_low, derphi_low, a_high, phi_high)

    a_j = T.switch(
        sequential_or(
            T.isnan(a_j),
            a_j > b - qchk,
            a_j < a + qchk
        ),
        a_low + asfloat(0.5) * dalpha,
        a_j
    )

    # Check new value of a_j
    phi_aj = phi(a_j)
    derphi_aj = derphi(a_j)

    cond1 = T.or_(
        phi_aj > (phi0 + c1 * a_j * derphi0),
        phi_aj >= phi_low
    )
    cond2 = derphi_aj * (a_high - a_low) >= zero

    # Switches just make more sense here because they have a C
    # implementation and they get composed
    phi_rec = ifelse(cond1, phi_high, T.switch(cond2, phi_high, phi_low))
    a_rec = ifelse(cond1, a_high, T.switch(cond2, a_high, a_low))
    a_high = ifelse(cond1, a_j, T.switch(cond2, a_low, a_high))
    phi_high = ifelse(cond1, phi_aj, T.switch(cond2, phi_low, phi_high))

    onlyif = sequential_and(phi_aj <= phi0 + c1 * a_j * derphi0,
                            phi_aj < phi_low,
                            abs(derphi_aj) <= -c2 * derphi0)

    a_low = T.switch(cond1, a_low, a_j)
    phi_low = T.switch(cond1, phi_low, phi_aj)
    derphi_low = ifelse(cond1, derphi_low, derphi_aj)
    vderphi_aj = ifelse(cond1, nan, T.switch(cond2, derphi_aj, nan))
    states = [phi_rec, a_rec, a_low, a_high, phi_high, phi_low,
              derphi_low, zero, zero, zero]

    outs, updates = theano.scan(
        zoom_itertion_step,
        outputs_info=states,
        n_steps=maxiter
    )

    a_star = ifelse(onlyif, a_j, outs[7][-1])
    val_star = ifelse(onlyif, phi_aj, outs[8][-1])
    valprime = ifelse(onlyif, vderphi_aj, outs[9][-1])

    ## WARNING !! I ignore updates given by scan which I should not do !!!
    return a_star, val_star, valprime
