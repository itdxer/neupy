import theano
import theano.tensor as T
from theano.ifelse import ifelse
import numpy as np

from neupy.core.properties import (ChoiceProperty, NumberProperty,
                                   WithdrawProperty)
from neupy.algorithms.gd import StepSelectionBuiltIn
from neupy.algorithms.utils import parameter_values, setup_parameter_updates
from neupy.optimizations.wolfe import line_search
from neupy.layers.utils import count_parameters, iter_parameters
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


class QuasiNewton(StepSelectionBuiltIn, GradientDescent):
    """
    Quasi-Newton algorithm optimization.

    Parameters
    ----------
    update_function : {{'bfgs', 'dfp', 'psb', 'sr1'}}
        Update function. Defaults to ``bfgs``.

    h0_scale : float
        Default Hessian matrix is an identity matrix. The
        ``h0_scale`` parameter scales identity matrix.
        Defaults to ``1``.

    {GradientDescent.connection}

    {GradientDescent.error}

    {GradientDescent.show_epoch}

    {GradientDescent.shuffle_data}

    {GradientDescent.epoch_end_signal}

    {GradientDescent.train_end_signal}

    {GradientDescent.verbose}

    {GradientDescent.addons}

    Attributes
    ----------
    {GradientDescent.Attributes}

    Methods
    -------
    {GradientDescent.Methods}

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms
    >>>
    >>> x_train = np.array([[1, 2], [3, 4]])
    >>> y_train = np.array([[1], [0]])
    >>>
    >>> qnnet = algorithms.QuasiNewton(
    ...     (2, 3, 1),
    ...     update_function='bfgs'
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

    step = WithdrawProperty()

    def init_variables(self):
        super(QuasiNewton, self).init_variables()
        n_params = count_parameters(self.connection)
        self.variables.update(
            inv_hessian=theano.shared(
                name='algo:quasi-newton/matrix:inv-hessian',
                value=asfloat(self.h0_scale * np.eye(int(n_params))),
            ),
            prev_params=theano.shared(
                name='algo:quasi-newton/vector:prev-params',
                value=asfloat(np.zeros(n_params)),
            ),
            prev_full_gradient=theano.shared(
                name='algo:quasi-newton/vector:prev-full-gradient',
                value=asfloat(np.zeros(n_params)),
            ),
        )

    def init_train_updates(self):
        network_inputs = self.variables.network_inputs
        network_output = self.variables.network_output
        inv_hessian = self.variables.inv_hessian
        prev_params = self.variables.prev_params
        prev_full_gradient = self.variables.prev_full_gradient

        params = parameter_values(self.connection)
        param_vector = T.concatenate([param.flatten() for param in params])

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
        layers_and_parameters = list(iter_parameters(self.layers))

        def prediction(step):
            updated_params = param_vector + step * param_delta

            # This trick allow us to replace shared variables
            # with theano variables and get output from the network
            start_pos = 0
            for layer, attrname, param in layers_and_parameters:
                end_pos = start_pos + param.size
                updated_param_value = T.reshape(
                    updated_params[start_pos:end_pos],
                    param.shape
                )
                setattr(layer, attrname, updated_param_value)
                start_pos = end_pos

            output = self.connection.output(*network_inputs)

            # Restore previous parameters
            for layer, attrname, param in layers_and_parameters:
                setattr(layer, attrname, param)

            return output

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
