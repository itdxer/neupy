from operator import mul

import theano
import theano.tensor as T
from theano.ifelse import ifelse
import numpy as np

from neupy.core.properties import (ChoiceProperty, ProperFractionProperty,
                                   NumberProperty)
from neupy.algorithms.utils import parameters2vector, iter_parameters
from neupy.utils import asfloat
from .base import GradientDescent


__all__ = ('QuasiNewton',)


def line_search(amax, c1=1e-5, c2=0.9):
    """ Line search method that satisfied Wolfe conditions

    Parameters
    ----------
    c1 : float
        Parameter for Armijo condition rule.
    c2 : float
        Parameter for curvature condition rule.
    amax : float
        Upper bound for value a.

    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions.  See Wright and Nocedal, 'Numerical Optimization',
    1999, pg. 59-60.
    """

    if not 0 < c1 < 1:
        raise ValueError("c1 should be a float between 0 and 1")

    if not 0 < c2 < 1:
        raise ValueError("c2 should be a float between 0 and 1")

    if c2 < c1:
        raise ValueError("c2 needs to be greater than c1")

    if amax <= 0:
        raise ValueError("amax needs to be greater than 0")


one = T.constant(np.asarray(1, dtype=theano.config.floatX))
zero = T.constant(np.asarray(0, dtype=theano.config.floatX))
nan = T.constant(np.asarray(np.nan, dtype=theano.config.floatX))

true = T.constant(np.asarray(1, dtype='int8'))
false = T.constant(np.asarray(0, dtype='int8'))


def scalar_search_wolfe2(phi,
                         derphi,
                         phi0=None,
                         old_phi0=None,
                         derphi0=None,
                         n_iters=20,
                         c1=1e-4,
                         c2=0.9,
                        profile=False):
    """
    Find alpha that satisfies strong Wolfe conditions.
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
    profile : flag (boolean)
        True if you want printouts of profiling information
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
    else:
        phi0 = phi0

    if derphi0 is None and derphi is not None:
        derphi0 = derphi(zero)
    else:
        derphi0 = derphi0

    alpha0 = zero
    alpha0.name = 'alpha0'
    if old_phi0 is not None:
        alpha1 = T.minimum(one,
                            np.asarray(1.01, dtype=theano.config.floatX) *
                            np.asarray(2, dtype=theano.config.floatX) * \
                            (phi0 - old_phi0) / derphi0)
    else:
        old_phi0 = nan
        alpha1 = one

    alpha1 = T.switch(alpha1 < zero, one, alpha1)
    alpha1.name = 'alpha1'

    # This shouldn't happen. Perhaps the increment has slipped below
    # machine precision?  For now, set the return variables skip the
    # useless while loop, and raise warnflag=2 due to possible imprecision.
    phi0 = T.switch(T.eq(alpha1, zero), old_phi0, phi0)
    # I need a lazyif for alpha1 == 0 !!!
    phi_a1 = ifelse(T.eq(alpha1, zero), phi0,
                    phi(alpha1), name='phi_a1')
    phi_a1.name = 'phi_a1'

    phi_a0 = phi0
    phi_a0.name = 'phi_a0'
    derphi_a0 = derphi0
    derphi_a0.name = 'derphi_a0'
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
                _zoom(alpha0, alpha1, phi_a0, phi_a1, derphi_a0,
                      phi, derphi, phi0, derphi0, c1, c2,
                     profile=profile)
        alpha_star_c3, phi_star_c3, derphi_star_c3 = \
                _zoom(alpha1, alpha0, phi_a1, phi_a0, derphi_a1, phi,
                      derphi, phi0, derphi0, c1, c2,
                     profile=profile)
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
                 ifelse(lazy_or('allconds',
                                cond1,
                                cond2,
                                cond3),
                        phi_a1,
                        nw_phi,
                        name='nwphi1'),
                 ifelse(cond1, derphi_a0, derphi_a1, name='derphi'),
                 i_t + one,
                 alpha_star,
                 phi_star,
                 derphi_star],
                theano.scan_module.scan_utils.until(
                    lazy_or('until_cond_',
                            T.eq(nw_alpha1, zero),
                            cond1,
                            cond2,
                            cond3)))
    states = [alpha0, alpha1, phi_a0, phi_a1, derphi_a0]
    # i_t
    states.append(zero)
    # alpha_star
    states.append(zero)
    # phi_star
    states.append(zero)
    # derphi_star
    states.append(zero)
    # print 'while_search'
    outs, updates = theano.scan(while_search,
                         outputs_info=states,
                         n_steps=maxiter,
                         name='while_search',
                         mode=theano.Mode(linker='cvm_nogc'),
                         profile=profile)
    # print 'done_while_search'
    out3 = outs[-3][-1]
    out2 = outs[-2][-1]
    out1 = outs[-1][-1]
    alpha_star, phi_star, derphi_star = \
            ifelse(T.eq(alpha1, zero),
                        (nan, phi0, nan),
                        (out3, out2, out1), name='main_alphastar')
    return alpha_star, phi_star,  phi0, derphi_star


def lazy_or(name='none', *args):
    """
    .. todo::
        WRITEME
    """
    def apply_me(args):
        if len(args) == 1:
            return args[0]
        else:
            rval = ifelse(args[0], true, apply_me(args[1:]),
                          name=name + str(len(args)))
            return rval
    return apply_me(args)


def _quadmin(a, fa, fpa, b, fb):
    """
    Finds the minimizer for a quadratic polynomial that goes through
    the points (a,fa), (b,fb) with derivative at a of fpa.
    Parameters
    ----------
    a : WRITEME
    fa : WRITEME
    fpa : WRITEME
    b : WRITEME
    fb : WRITEME
    Returns
    -------
    WRITEME
    """
    # f(x) = B*(x-a)^2 + C*(x-a) + D
    D = fa
    C = fpa
    db = b - a * one

    B = (fb - D - C * db) / (db * db)
    # Note : `lazy if` would make more sense, but it is not
    #        implemented in C right now
    # lazy_or('quadmin',T.eq(db , zero), (B <= zero)),
    # xmin = T.switch(T.bitwise_or(T.eq(db,zero), B <= zero),
    xmin = T.switch(lazy_or(T.eq(db, zero), B <= zero),
                     nan,
                     a - C /\
                     (np.asarray(2, dtype=theano.config.floatX) * B))
    return xmin


def lazy_and(name='node', *args):
    """
    .. todo::
        WRITEME
    """
    def apply_me(args):
        if len(args) == 1:
            return args[0]
        else:
            rval = ifelse(T.eq(args[0], zero), false, apply_me(args[1:]),
                         name=name + str(len(args)))
            return rval
    return apply_me(args)


def _cubicmin(a, fa, fpa, b, fb, c, fc):
    """
    Finds the minimizer for a cubic polynomial that goes through the
    points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.
    If no minimizer can be found return None
    Parameters
    ----------
    a : WRITEME
    fa : WRITEME
    fpa : WRITEME
    b : WRITEME
    fb : WRITEME
    c : WRITEME
    fc : WRITEME
    Returns
    -------
    WRITEME
    """
    # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D
    a.name = 'a'
    fa.name = 'fa'
    fpa.name = 'fpa'
    fb.name = 'fb'
    fc.name = 'fc'
    C = fpa
    D = fa
    db = b - a
    dc = c - a

    denom = (db * dc) ** 2 * (db - dc)
    d1_00 = dc ** 2
    d1_01 = -db ** 2
    d1_10 = -dc ** 3
    d1_11 = db ** 3
    t1_0 = fb - fa - C * db
    t1_1 = fc - fa - C * dc
    A = d1_00 * t1_0 + d1_01 * t1_1
    B = d1_10 * t1_0 + d1_11 * t1_1
    A /= denom
    B /= denom
    radical = B * B - 3 * A * C
    radical.name = 'radical'
    db.name = 'db'
    dc.name = 'dc'
    b.name = 'b'
    c.name = 'c'
    A.name = 'A'
    #cond = T.bitwise_or(radical < zero,
    #       T.bitwise_or(T.eq(db,zero),
    #       T.bitwise_or(T.eq(dc,zero),
    #       T.bitwise_or(T.eq(b, c),
    #                    T.eq(A, zero)))))

    cond = lazy_or('cubicmin',
                   radical < zero,
                   T.eq(db, zero),
                   T.eq(dc, zero),
                   T.eq(b, c),
                   T.eq(A, zero))
    # Note: `lazy if` would make more sense, but it is not
    #       implemented in C right now
    xmin = T.switch(cond, constant(np.nan),
                         a + (-B + T.sqrt(radical)) / (3 * A))
    return xmin


def my_not(arg):
    """
    .. todo::
        WRITEME
    """
    return T.eq(arg, zero)


def constant(value):
    """
    .. todo::
        WRITEME
    """
    return T.constant(np.asarray(value, dtype=theano.config.floatX))


def _zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo,
          phi, derphi, phi0, derphi0, c1, c2,
          n_iters=10,
          profile=False):
    """
    WRITEME
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
    profile : bool
        True if you want printouts of profiling information
    """
    # Function reprensenting the computations of one step of the while loop
    def while_zoom(phi_rec, a_rec, a_lo, a_hi, phi_hi,
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
        a_j_cubic = _cubicmin(a_lo, phi_lo, derphi_lo,
                              a_hi, phi_hi, a_rec, phi_rec)
        # quadric interpolation
        qchk = delta2 * dalpha
        a_j_quad = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
        cond_q = lazy_or('condq',
                         T.isnan(a_j_quad),
                         a_j_quad > b - qchk,
                         a_j_quad < a + qchk)
        a_j_quad = T.switch(cond_q, a_lo +
                             np.asarray(0.5, dtype=theano.config.floatX) * \
                             dalpha, a_j_quad)

        # pick between the two ..
        cond_c = lazy_or('condc',
                         T.isnan(a_j_cubic),
                         T.bitwise_or(a_j_cubic > b - cchk,
                                       a_j_cubic < a + cchk))
        # this lazy if actually decides if we need to run the quadric
        # interpolation
        a_j = T.switch(cond_c, a_j_quad, a_j_cubic)
        #a_j = ifelse(cond_c, a_j_quad,  a_j_cubic)

        # Check new value of a_j
        phi_aj = phi(a_j)
        derphi_aj = derphi(a_j)

        stop = lazy_and('stop',
                        T.bitwise_and(phi_aj <= phi0 + c1 * a_j * derphi0,
                                       phi_aj < phi_lo),
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
    a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
    cond_q = lazy_or('mcond_q',
                     T.isnan(a_j),
                     T.bitwise_or(a_j > b - qchk,
                                   a_j < a + qchk))

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

    onlyif = lazy_and('only_if',
                      T.bitwise_and(phi_aj <= phi0 + c1 * a_j * derphi0,
                                     phi_aj < phi_lo),
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
    states = [phi_rec, a_rec, a_lo, a_hi, phi_hi, phi_lo, derphi_lo, zero, zero, zero]

    # print'while_zoom'
    outs, updates = theano.scan(while_zoom,
                         outputs_info=states,
                         n_steps=maxiter,
                         name='while_zoom',
                         mode=theano.Mode(linker='cvm_nogc'),
                         profile=profile)
    # print 'done_while'
    a_star = ifelse(onlyif, a_j, outs[7][-1], name='astar')
    val_star = ifelse(onlyif, phi_aj, outs[8][-1], name='valstar')
    valprime = ifelse(onlyif, vderphi_aj, outs[9][-1], name='valprime')

    ## WARNING !! I ignore updates given by scan which I should not do !!!
    return a_star, val_star, valprime


def bfgs(inverse_hessian, weight_delta, gradient_delta, maxrho=1e4):
    ident_matrix = T.eye(inverse_hessian.shape[0])

    rho = 1. / gradient_delta.dot(weight_delta)
    rho = ifelse(
        T.isinf(rho),
        maxrho * T.sgn(rho),
        rho,
    )

    param1 = ident_matrix - T.outer(weight_delta, gradient_delta) * rho
    param2 = ident_matrix - T.outer(gradient_delta, weight_delta) * rho
    param3 = rho * T.outer(weight_delta, weight_delta)

    return param1.dot(inverse_hessian).dot(param2) + param3


def dfp(inverse_hessian, weight_delta, gradient_delta, maxnum=1e5):
    quasi_dot_gradient = inverse_hessian.dot(gradient_delta)

    param1 = (
        T.outer(weight_delta, weight_delta)
    ) / (
        T.dot(gradient_delta, weight_delta)
    )
    param2_numerator = T.clip(
        T.outer(quasi_dot_gradient, gradient_delta) * inverse_hessian,
        -maxnum, maxnum
    )
    param2_denominator = gradient_delta.dot(quasi_dot_gradient)
    param2 = param2_numerator / param2_denominator

    return inverse_hessian + param1 - param2


def psb(inverse_hessian, weight_delta, gradient_delta, **options):
    gradient_delta_t = gradient_delta.T
    param = weight_delta - inverse_hessian.dot(gradient_delta)

    devider = (1. / T.dot(gradient_delta, gradient_delta))
    param1 = T.outer(param, gradient_delta) + T.outer(gradient_delta, param)
    param2 = (
        T.dot(gradient_delta, param) *
        T.outer(gradient_delta, gradient_delta_t)
    )

    return inverse_hessian + param1 * devider - param2 * devider ** 2


def sr1(inverse_hessian, weight_delta, gradient_delta, epsilon=1e-8):
    param = weight_delta - inverse_hessian.dot(gradient_delta)
    denominator = T.dot(param, gradient_delta)

    return ifelse(
        T.lt(
            T.abs_(denominator),
            epsilon * param.norm(L=2) * gradient_delta.norm(L=2)
        ),
        inverse_hessian,
        inverse_hessian + T.outer(param, param) / denominator
    )


class QuasiNewton(GradientDescent):
    """ Quasi-Newton algorithm optimization.

    Parameters
    ----------
    {GradientDescent.optimizations}
    {ConstructableNetwork.connection}
    {SupervisedConstructableNetwork.error}
    {BaseNetwork.step}
    {BaseNetwork.show_epoch}
    {BaseNetwork.shuffle_data}
    {BaseNetwork.epoch_end_signal}
    {BaseNetwork.train_end_signal}
    {Verbose.verbose}

    Methods
    -------
    {BaseSkeleton.predict}
    {SupervisedLearning.train}
    {BaseSkeleton.fit}
    {BaseNetwork.plot_errors}
    {BaseNetwork.last_error}
    {BaseNetwork.last_validation_error}
    {BaseNetwork.previous_error}

    Examples
    --------
    Simple example

    >>> import numpy as np
    >>> from neupy import algorithms
    >>>
    >>> x_train = np.array([[1, 2], [3, 4]])
    >>> y_train = np.array([[1], [0]])
    >>>
    >>> qnnet = algorithms.QuasiNewton(
    ...     (2, 3, 1),
    ...     update_function='bfgs',
    ...     verbose=False
    ... )
    >>> qnnet.train(x_train, y_train)

    See Also
    --------
    :network:`GradientDescent` : GradientDescent algorithm.
    """
    update_function = ChoiceProperty(
        default='bfgs',
        choices={
            'bfgs': bfgs,
            'dfp': dfp,
            'psb': psb,
            'sr1': sr1,
        }
    )
    h0_scale = NumberProperty(default=1, minval=0)
    gradient_tol = ProperFractionProperty(default=1e-5)

    def init_variables(self):
        super(QuasiNewton, self).init_variables()
        n_params = sum(p.get_value().size for p in iter_parameters(self))
        self.variables.update(
            inv_hessian=theano.shared(
                name='inv_hessian',
                value=asfloat(self.h0_scale * np.eye(int(n_params))),
            ),
            prev_params=theano.shared(
                name='prev_params',
                value=asfloat(np.zeros(n_params)),
            ),
            prev_grads=theano.shared(
                name='prev_grads',
                value=asfloat(np.zeros(n_params)),
            ),
        )

    def init_train_updates(self):
        network_input = self.variables.network_input
        network_output = self.variables.network_output
        inv_hessian = self.variables.inv_hessian
        prev_params = self.variables.prev_params
        prev_grads = self.variables.prev_grads

        params = list(iter_parameters(self))
        param_vector = parameters2vector(self)

        grads = []
        n_params = 0
        for param in params:
            gradient = T.grad(self.variables.error_func, wrt=param)
            grads.append(gradient.flatten())
            n_params += param.size

        grads = T.concatenate(grads)

        new_inv_hessian = ifelse(
            T.eq(self.variables.epoch, 1),
            inv_hessian,
            self.update_function(inv_hessian,
                                 param_vector - prev_params,
                                 grads - prev_grads)
        )
        param_delta = -new_inv_hessian.dot(grads)

        # from pylearn2.optimization.linesearch import scalar_search_wolfe2
        def prediction(step):
            updated_params = param_vector + step * param_delta

            layer_input = network_input
            start_pos = 0
            for layer in self.train_layers:
                for param in layer.parameters:
                    end_pos = start_pos + param.size
                    setattr(layer, param.name.split('_')[0], T.reshape(
                        updated_params[start_pos:end_pos],
                        param.shape
                    ))
                    start_pos = end_pos
                layer_input = layer.output(layer_input)
            return layer_input

        def phi(step):
            return self.error(prediction(step), network_output)

        def derphi(step):
            error_func = self.error(prediction(step), network_output)
            return T.grad(error_func, wrt=step)

        rvals = scalar_search_wolfe2(phi, derphi)

        updated_params = param_vector + rvals[0] * param_delta

        start_pos = 0
        updates = []
        for param in params:
            end_pos = start_pos + param.size
            updates.append((
                param,
                T.reshape(
                    updated_params[start_pos:end_pos],
                    param.shape
                )
            ))
            start_pos = end_pos

        updates.extend([
            (inv_hessian, new_inv_hessian),
            (prev_params, param_vector),
            (prev_grads, grads),
        ])

        return updates
