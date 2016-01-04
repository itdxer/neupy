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
                         n_iters=20, c1=1e-4, c2=0.9):
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
    maxiter = n_iters

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
        Output from function ``y`` at some point ``a``.
    y_prime_a : float or theano variable
        Output from function ``y'`` (``y`` derivative) at some
        point ``a``.
    x_b : float or theano variable
        Right point ``a`` in the ``x`` axis.
    y_b : float or theano variable
        Output from function ``y`` at some point ``b``.

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
    y_a : float or theano variable
    y_prime_a : float or theano variable
    x_b : float or theano variable
    y_b : float or theano variable
    x_c : float or theano variable
    y_c : float or theano variable

    Returns
    -------
    object
        Theano variable that after evaluation is equal to
        point ``x`` which is minimizer for cubic function.
    """

    db = x_b - x_a
    dc = x_c - x_a

    denom = (db * dc) ** 2 * (db - dc)
    t1_0 = y_b - y_a - y_prime_a * db
    t1_1 = y_c - y_a - y_prime_a * dc

    A = (dc ** 2 * t1_0 - db ** 2 * t1_1) / denom
    B = (-dc ** 3 * t1_0 + db ** 3 * t1_1) / denom
    radical = B ** 2 - 3 * A * y_prime_a

    return T.switch(
        sequential_or(
            radical < zero,
            T.eq(db, zero),
            T.eq(dc, zero),
            T.eq(x_b, x_c),
            T.eq(A, zero)
        ),
        np.nan,
        x_a + (-B + T.sqrt(radical)) / (3 * A)
    )


def zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo,
          phi, derphi, phi0, derphi0, c1, c2, n_iters=10):
    """
    Notes
    -----
    Part of the optimization algorithm in `scalar_search_wolfe2`.

    Parameters
    ----------
    a_lo : float
        Step size
    a_hi : float
        Step size
    phi_lo : float
        Value of f at a_lo
    phi_hi : float
        Value of f at a_hi
    derphi_lo : float
        Value of derivative at a_lo
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
    # Function reprensenting the computations of one step of the while loop
    def whilezoom(phi_rec, a_rec, a_lo, a_hi, phi_hi,
                   phi_lo, derphi_lo, a_star, val_star, valprime):
        # interpolate to find a trial step length between a_lo and
        # a_hi Need to choose interpolation here.  Use cubic
        # interpolation and then if the result is within delta *
        # dalpha or outside of the interval bounded by a_lo or a_hi
        # then use quadratic interpolation, if the result is still too
        # close, then use bisection
        dalpha = a_hi - a_lo
        a = T.switch(dalpha < zero, a_hi, a_lo)
        b = T.switch(dalpha < zero, a_lo, a_hi)

        # minimizer of cubic interpolant
        # (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
        #
        # if the result is too close to the end points (or out of the
        # interval) then use quadratic interpolation with phi_lo,
        # derphi_lo and phi_hi if the result is stil too close to the
        # end points (or out of the interval) then use bisection

        # cubic interpolation
        cchk = delta1 * dalpha
        a_j_cubic = cubic_minimizer(a_lo, phi_lo, derphi_lo,
                                    a_hi, phi_hi, a_rec, phi_rec)
        # quadric interpolation
        qchk = delta2 * dalpha
        a_j_quad = quadratic_minimizer(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
        cond_q = sequential_or(T.isnan(a_j_quad),
                         a_j_quad > b - qchk,
                         a_j_quad < a + qchk)
        a_j_quad = T.switch(cond_q, a_lo + asfloat(0.5) * dalpha, a_j_quad)

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
                              phi_aj < phi_lo,
                              abs(derphi_aj) <= -c2 * derphi0)

        cond1 = T.bitwise_or(phi_aj > phi0 + c1 * a_j * derphi0,
                              phi_aj >= phi_lo)
        cond2 = derphi_aj * (a_hi - a_lo) >= zero

        # Switches just make more sense here because they have a C
        # implementation and they get composed
        phi_rec = ifelse(cond1,
                         phi_hi,
                         T.switch(cond2, phi_hi, phi_lo),
                         name='phi_rec')
        a_rec = ifelse(cond1,
                       a_hi,
                       T.switch(cond2, a_hi, a_lo),
                         name='a_rec')
        a_hi = ifelse(cond1, a_j,
                      T.switch(cond2, a_lo, a_hi),
                      name='a_hi')
        phi_hi = ifelse(cond1, phi_aj,
                        T.switch(cond2, phi_lo, phi_hi),
                        name='phi_hi')

        a_lo = T.switch(cond1, a_lo, a_j)
        phi_lo = T.switch(cond1, phi_lo, phi_aj)
        derphi_lo = ifelse(cond1, derphi_lo, derphi_aj, name='derphi_lo')

        a_star = a_j
        val_star = phi_aj
        valprime = ifelse(cond1, nan,
                          T.switch(cond2, derphi_aj, nan), name='valprime')

        return ([phi_rec,
                 a_rec,
                 a_lo,
                 a_hi,
                 phi_hi,
                 phi_lo,
                 derphi_lo,
                 a_star,
                 val_star,
                 valprime],
                theano.scan_module.scan_utils.until(stop))

    maxiter = n_iters
    # cubic interpolant check
    delta1 = T.constant(np.asarray(0.2,
                                       dtype=theano.config.floatX))
    # quadratic interpolant check
    delta2 = T.constant(np.asarray(0.1,
                                       dtype=theano.config.floatX))
    phi_rec = phi0
    a_rec = zero

    # Initial iteration

    dalpha = a_hi - a_lo
    a = T.switch(dalpha < zero, a_hi, a_lo)
    b = T.switch(dalpha < zero, a_lo, a_hi)
    #a = ifelse(dalpha < 0, a_hi, a_lo)
    #b = ifelse(dalpha < 0, a_lo, a_hi)

    # minimizer of cubic interpolant
    # (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
    #
    # if the result is too close to the end points (or out of the
    # interval) then use quadratic interpolation with phi_lo,
    # derphi_lo and phi_hi if the result is stil too close to the
    # end points (or out of the interval) then use bisection

    # quadric interpolation
    qchk = delta2 * dalpha
    a_j = quadratic_minimizer(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
    cond_q = sequential_or(T.isnan(a_j),
                           a_j > b - qchk,
                           a_j < a + qchk)

    a_j = T.switch(cond_q, a_lo +
                    np.asarray(0.5, dtype=theano.config.floatX) * \
                    dalpha, a_j)

    # Check new value of a_j
    phi_aj = phi(a_j)
    derphi_aj = derphi(a_j)

    cond1 = T.bitwise_or(phi_aj > phi0 + c1 * a_j * derphi0,
                          phi_aj >= phi_lo)
    cond2 = derphi_aj * (a_hi - a_lo) >= zero

    # Switches just make more sense here because they have a C
    # implementation and they get composed
    phi_rec = ifelse(cond1,
                     phi_hi,
                     T.switch(cond2, phi_hi, phi_lo),
                     name='mphirec')
    a_rec = ifelse(cond1,
                   a_hi,
                   T.switch(cond2, a_hi, a_lo),
                   name='marec')
    a_hi = ifelse(cond1,
                  a_j,
                  T.switch(cond2, a_lo, a_hi),
                  name='mahi')
    phi_hi = ifelse(cond1,
                    phi_aj,
                    T.switch(cond2, phi_lo, phi_hi),
                    name='mphihi')

    onlyif = sequential_and(phi_aj <= phi0 + c1 * a_j * derphi0,
                            phi_aj < phi_lo,
                            abs(derphi_aj) <= -c2 * derphi0)

    a_lo = T.switch(cond1, a_lo, a_j)
    phi_lo = T.switch(cond1, phi_lo, phi_aj)
    derphi_lo = ifelse(cond1, derphi_lo, derphi_aj, name='derphi_lo_main')
    phi_rec.name = 'phi_rec'
    a_rec.name = 'a_rec'
    a_lo.name = 'a_lo'
    a_hi.name = 'a_hi'
    phi_hi.name = 'phi_hi'
    phi_lo.name = 'phi_lo'
    derphi_lo.name = 'derphi_lo'
    vderphi_aj = ifelse(cond1, nan, T.switch(cond2, derphi_aj, nan),
                        name='vderphi_aj')
    states = [phi_rec, a_rec, a_lo, a_hi, phi_hi, phi_lo,
              derphi_lo, zero, zero, zero]

    outs, updates = theano.scan(whilezoom,
                         outputs_info=states,
                         n_steps=maxiter,
                         name='whilezoom',
                        #  mode=theano.Mode(linker='cvm_nogc'),
                         )

    a_star = ifelse(onlyif, a_j, outs[7][-1], name='astar')
    val_star = ifelse(onlyif, phi_aj, outs[8][-1], name='valstar')
    valprime = ifelse(onlyif, vderphi_aj, outs[9][-1], name='valprime')

    ## WARNING !! I ignore updates given by scan which I should not do !!!
    return a_star, val_star, valprime
