import theano
import theano.tensor as T
from theano.ifelse import ifelse
import numpy as np

from neupy.core.properties import (ChoiceProperty, ProperFractionProperty,
                                   NumberProperty)
from neupy.algorithms.gd import NoStepSelection
from neupy.algorithms.utils import (parameters2vector, iter_parameters,
                                    setup_parameter_updates)
from neupy.optimizations.wolfe import line_search
from neupy.utils import asfloat
from .base import GradientDescent


__all__ = ('QuasiNewton',)


def bfgs(inverse_hessian, weight_delta, gradient_delta, maxrho=1e4):
    ident_matrix = T.eye(inverse_hessian.shape[0])

    maxrho = asfloat(maxrho)
    rho = asfloat(1.) / gradient_delta.dot(weight_delta)

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
    maxnum = asfloat(maxnum)
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
    epsilon = asfloat(epsilon)
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


class QuasiNewton(NoStepSelection, GradientDescent):
    """ Quasi-Newton algorithm optimization.

    Parameters
    ----------
    {GradientDescent.addons}
    {ConstructableNetwork.connection}
    {ConstructableNetwork.error}
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
    >>> qnnet.train(x_train, y_train, epochs=10)

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
            prev_full_gradient=theano.shared(
                name='prev_full_gradient',
                value=asfloat(np.zeros(n_params)),
            ),
        )

    def init_train_updates(self):
        network_input = self.variables.network_input
        network_output = self.variables.network_output
        inv_hessian = self.variables.inv_hessian
        prev_params = self.variables.prev_params
        prev_full_gradient = self.variables.prev_full_gradient

        params = list(iter_parameters(self))
        param_vector = parameters2vector(self)

        gradients = T.grad(self.variables.error_func, wrt=params)
        full_gradient = T.concatenate([grad.flatten() for grad in gradients])

        new_inv_hessian = ifelse(
            T.eq(self.variables.epoch, 1),
            inv_hessian,
            self.update_function(inv_hessian,
                                 param_vector - prev_params,
                                 full_gradient - prev_full_gradient)
        )
        param_delta = -new_inv_hessian.dot(full_gradient)

        def prediction(step):
            # TODO: I need to update this ugly solution later
            updated_params = param_vector + step * param_delta

            layer_input = network_input
            start_pos = 0
            for layer in self.layers:
                for param in layer.parameters:
                    end_pos = start_pos + param.size
                    parameter_name, parameter_id = param.name.split('_')
                    setattr(layer, parameter_name, T.reshape(
                        updated_params[start_pos:end_pos],
                        param.shape
                    ))
                    start_pos = end_pos
                layer_input = layer.output(layer_input)
            return layer_input

        def phi(step):
            return self.error(network_output, prediction(step))

        def derphi(step):
            error_func = self.error(network_output, prediction(step))
            return T.grad(error_func, wrt=step)

        step = asfloat(line_search(phi, derphi))
        updated_params = param_vector + step * param_delta
        updates = setup_parameter_updates(params, updated_params)

        updates.extend([
            (inv_hessian, new_inv_hessian),
            (prev_params, param_vector),
            (prev_full_gradient, full_gradient),
        ])

        return updates
