import tensorflow as tf

from neupy.core.properties import (ChoiceProperty, NumberProperty,
                                   WithdrawProperty)
from neupy.algorithms.gd import StepSelectionBuiltIn
from neupy.algorithms.utils import parameter_values, setup_parameter_updates
from neupy.optimizations.wolfe import line_search
from neupy.layers.utils import count_parameters, iter_parameters
from neupy.utils import asfloat, flatten, dot, outer, get_variable_size
from .base import GradientDescent


__all__ = ('QuasiNewton',)


def bfgs(inverse_hessian, weight_delta, gradient_delta, maxrho=1e4):
    with tf.name_scope('bfgs'):
        ident_matrix = tf.eye(int(inverse_hessian.shape[0]))
        rho = asfloat(1.) / dot(gradient_delta, weight_delta)

        rho = tf.where(
            tf.is_inf(rho),
            maxrho * tf.sign(rho),
            rho,
        )

        param1 = ident_matrix - outer(weight_delta, gradient_delta) * rho
        param2 = ident_matrix - outer(gradient_delta, weight_delta) * rho
        param3 = rho * outer(weight_delta, weight_delta)

        return dot(param1, tf.matmul(inverse_hessian, param2)) + param3


def dfp(inverse_hessian, weight_delta, gradient_delta, maxnum=1e5):
    with tf.name_scope('dfp'):
        quasi_dot_gradient = dot(inverse_hessian, gradient_delta)

        param1 = (
            outer(weight_delta, weight_delta)
        ) / (
            dot(gradient_delta, weight_delta)
        )
        param2_numerator = tf.clip_by_value(
            outer(quasi_dot_gradient, gradient_delta) * inverse_hessian,
            -maxnum, maxnum
        )
        param2_denominator = dot(gradient_delta, quasi_dot_gradient)
        param2 = param2_numerator / param2_denominator

        return inverse_hessian + param1 - param2


def psb(inverse_hessian, weight_delta, gradient_delta, **options):
    with tf.name_scope('psb'):
        gradient_delta_t = tf.transpose(gradient_delta)
        param = weight_delta - dot(inverse_hessian, gradient_delta)

        devider = 1. / dot(gradient_delta, gradient_delta)
        param1 = outer(param, gradient_delta) + outer(gradient_delta, param)
        param2 = (
            dot(gradient_delta, param) *
            outer(gradient_delta, gradient_delta_t)
        )

        return inverse_hessian + param1 * devider - param2 * devider ** 2


def sr1(inverse_hessian, weight_delta, gradient_delta, epsilon=1e-8):
    """
    Symmetric rank 1 (SR1). Generates update for the inverse hessian
    matrix adding symmetric rank-1 matrix. It's possible that there is no
    rank 1 updates for the matrix and in this case update won't be applied
    and original inverse hessian will be returned.
    """
    with tf.name_scope('sr1'):
        epsilon = asfloat(epsilon)
        param = weight_delta - dot(inverse_hessian, gradient_delta)
        denominator = dot(param, gradient_delta)

        return tf.where(
            # This check protects from the cases when update
            # doesn't exist. It's possible that during certain
            # iteration there is no rank-1 update for the matrix.
            tf.less(
                tf.abs(denominator),
                epsilon * tf.norm(param) * tf.norm(gradient_delta)
            ),
            inverse_hessian,
            inverse_hessian + outer(param, param) / denominator
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
        n_parameters = count_parameters(self.connection)

        self.variables.update(
            inv_hessian=tf.Variable(
                asfloat(self.h0_scale) * tf.eye(n_parameters),
                name="quasi-newton/inv-hessian",
                dtype=tf.float32,
            ),
            prev_params=tf.Variable(
                tf.zeros([n_parameters]),
                name="quasi-newton/prev-params",
                dtype=tf.float32,
            ),
            prev_full_gradient=tf.Variable(
                tf.zeros([n_parameters]),
                name="quasi-newton/prev-full-gradient",
                dtype=tf.float32,
            ),
        )

    def init_train_updates(self):
        network_inputs = self.variables.network_inputs
        network_output = self.variables.network_output
        inv_hessian = self.variables.inv_hessian
        prev_params = self.variables.prev_params
        prev_full_gradient = self.variables.prev_full_gradient

        params = parameter_values(self.connection)
        param_vector = tf.concat([flatten(param) for param in params], axis=0)

        gradients = tf.gradients(self.variables.error_func, params)
        full_gradient = tf.concat(
            [flatten(grad) for grad in gradients], axis=0)

        new_inv_hessian = tf.where(
            tf.equal(self.variables.epoch, 1),
            inv_hessian,
            self.update_function(
                inv_hessian,
                param_vector - prev_params,
                full_gradient - prev_full_gradient,
            )
        )
        param_delta = -dot(new_inv_hessian, full_gradient)
        layers_and_parameters = list(iter_parameters(self.layers))

        def prediction(step):
            step = asfloat(step)
            updated_params = param_vector + step * param_delta

            # This trick allow us to replace shared variables
            # with tensorflow variables and get output from the network
            start_pos = 0
            for layer, attrname, param in layers_and_parameters:
                end_pos = start_pos + get_variable_size(param)
                updated_param_value = tf.reshape(
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
            gradient, = tf.gradients(error_func, step)
            return gradient

        step = asfloat(line_search(phi, derphi))
        updated_params = param_vector + step * param_delta
        updates = setup_parameter_updates(params, updated_params)

        updates.extend([
            (inv_hessian, new_inv_hessian),
            (prev_params, param_vector),
            (prev_full_gradient, full_gradient),
        ])

        return updates
