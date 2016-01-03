from operator import mul

import theano
import theano.tensor as T
from theano.ifelse import ifelse
import numpy as np

from neupy.core.properties import (ChoiceProperty, ProperFractionProperty,
                                   NumberProperty)
from neupy.algorithms.utils import parameters2vector, iter_parameters
from neupy.optimizations.wolfe import scalar_search_wolfe2
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
