import tensorflow as tf

from neupy.core.config import Configurable
from neupy.core.properties import (
    ChoiceProperty, NumberProperty,
    WithdrawProperty, IntProperty,
)
from neupy.utils.tf_utils import setup_parameter_updates
from neupy.algorithms.minsearch.wolfe import line_search
from neupy.utils import (
    asfloat, dot, outer, as_tuple,
    function_name_scope, make_single_vector,
)
from .base import BaseOptimizer


__all__ = ('QuasiNewton',)


class WolfeLineSearchForStep(Configurable):
    """
    Class that has all functions required in order to apply line search over
    step parameter that used during the network training.

    Parameters
    ----------
    wolfe_maxiter : int
        Controls maximum number of iteration during the line search that
        identifies optimal step size during the weight update stage.
        Defaults to ``20``.

    wolfe_c1 : float
        Parameter for Armijo condition rule. It's used during the line search
        that identifies optimal step size during the weight update stage.
        Defaults ``1e-4``.

    wolfe_c2 : float
        Parameter for curvature condition rule. It's used during the line
        search that identifies optimal step size during the weight update
        stage. Defaults ``0.9``.
    """
    wolfe_maxiter = IntProperty(default=20, minval=0)
    wolfe_c1 = NumberProperty(default=1e-4, minval=0)
    wolfe_c2 = NumberProperty(default=0.9, minval=0)

    def find_optimal_step(self, parameter_vector, parameter_update):
        network_inputs = as_tuple(self.network.inputs)
        layers_and_parameters = list(self.network.variables.items())

        def prediction(step):
            step = asfloat(step)
            updated_params = parameter_vector + step * parameter_update

            try:
                # This trick allow us to replace shared variables
                # with tensorflow variables and get output from the network
                start_pos = 0
                for (layer, varname), variable in layers_and_parameters:
                    n_param_values = int(variable.shape.num_elements())
                    end_pos = start_pos + n_param_values

                    updated_param_value = tf.reshape(
                        updated_params[start_pos:end_pos],
                        variable.shape)

                    setattr(layer, varname, updated_param_value)
                    start_pos = end_pos

                output = self.network.output(*network_inputs, training=True)

            finally:
                # Restore previous parameters
                for (layer, varname), param in layers_and_parameters:
                    setattr(layer, varname, param)

            return output

        def phi(step):
            return self.loss(self.target, prediction(step))

        def derphi(step):
            error_func = self.loss(self.target, prediction(step))
            gradient, = tf.gradients(error_func, step)
            return gradient

        return line_search(
            phi, derphi, self.wolfe_maxiter,
            self.wolfe_c1, self.wolfe_c2)


@function_name_scope
def safe_reciprocal(value, epsilon):
    """
    The same as regular function in the tensorflow accept that it ensures
    that non of the input values have magnitude smaller than epsilon.
    Otherwise small values will be capped to the epsilon.
    """
    inv_epsilon = 1. / epsilon
    return tf.clip_by_value(
        tf.reciprocal(value),
        -inv_epsilon,
        inv_epsilon
    )


@function_name_scope
def safe_division(numerator, denominator, epsilon):
    """
    The same as regular function in the tensorflow accept that it ensures
    that non of the denominator values have magnutide smaller than epsilon.
    Otherwise small values will be capped to the epsilon.
    """
    inv_denominator = safe_reciprocal(denominator, epsilon)
    return numerator * inv_denominator


@function_name_scope
def bfgs(inv_H, delta_w, delta_grad, epsilon=1e-7):
    """
    It can suffer from round-off error and inaccurate line searches.
    """
    n_parameters = int(inv_H.shape[0])

    I = tf.eye(n_parameters)
    rho = safe_reciprocal(dot(delta_grad, delta_w), epsilon)

    X = I - outer(delta_w, delta_grad) * rho
    X_T = tf.transpose(X)
    Z = rho * outer(delta_w, delta_w)

    return tf.matmul(X, tf.matmul(inv_H, X_T)) + Z


@function_name_scope
def dfp(inv_H, delta_w, delta_grad, epsilon=1e-7):
    """
    DFP is a method very similar to BFGS. It's rank 2 formula update.
    It can suffer from round-off error and inaccurate line searches.
    """
    inv_H_dot_grad = dot(inv_H, delta_grad)

    x = safe_division(
        outer(delta_w, delta_w),
        dot(delta_grad, delta_w),
        epsilon
    )
    y = safe_division(
        tf.matmul(outer(inv_H_dot_grad, delta_grad), inv_H),
        dot(delta_grad, inv_H_dot_grad),
        epsilon
    )

    return inv_H - y + x


@function_name_scope
def sr1(inv_H, delta_w, delta_grad, epsilon=1e-7):
    """
    Symmetric rank 1 (SR1). Generates update for the inverse hessian
    matrix adding symmetric rank-1 matrix. It's possible that there is no
    rank 1 updates for the matrix and in this case update won't be applied
    and original inverse hessian will be returned.
    """
    param = delta_w - dot(inv_H, delta_grad)
    denominator = dot(param, delta_grad)

    return tf.where(
        # This check protects from the cases when update
        # doesn't exist. It's possible that during certain
        # iteration there is no rank-1 update for the matrix.
        tf.less(
            tf.abs(denominator),
            epsilon * tf.norm(param) * tf.norm(delta_grad)
        ),
        inv_H,
        inv_H + outer(param, param) / denominator
    )


class QuasiNewton(WolfeLineSearchForStep, BaseOptimizer):
    """
    Quasi-Newton algorithm. Every iteration quasi-Network method approximates
    inverse Hessian matrix with iterative updates. It doesn't have ``step``
    parameter. Instead, algorithm applies line search for the step parameter
    that satisfies strong Wolfe condition. Parameters that control wolfe
    search start with the ``wolfe_`` prefix.

    Parameters
    ----------
    update_function : ``bfgs``, ``dfp``, ``sr1``
        Update function for the iterative inverse hessian matrix
        approximation. Defaults to ``bfgs``.

        - ``bfgs`` -  It's rank 2 formula update. It can suffer from
          round-off error and inaccurate line searches.

        - ``dfp`` - DFP is a method very similar to BFGS. It's rank 2 formula
          update. It can suffer from round-off error and inaccurate line
          searches.

        - ``sr1`` - Symmetric rank 1 (SR1). Generates update for the
          inverse hessian matrix adding symmetric rank-1 matrix. It's
          possible that there is no rank 1 updates for the matrix and in
          this case update won't be applied and original inverse hessian
          will be returned.

    h0_scale : float
        Default Hessian matrix is an identity matrix. The
        ``h0_scale`` parameter scales identity matrix.
        Defaults to ``1``.

    epsilon : float
        Controls numerical stability for the ``update_function`` parameter.
        Defaults to ``1e-7``.

    {WolfeLineSearchForStep.Parameters}

    {BaseOptimizer.network}

    {BaseOptimizer.loss}

    {BaseOptimizer.show_epoch}

    {BaseOptimizer.shuffle_data}

    {BaseOptimizer.signals}

    {BaseOptimizer.verbose}

    {BaseOptimizer.regularizer}

    Notes
    -----
    - Method requires all training data during propagation, which means
      it cannot be trained with mini-batches.

    Attributes
    ----------
    {BaseOptimizer.Attributes}

    Methods
    -------
    {BaseOptimizer.Methods}

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms
    >>> from neupy.layers import *
    >>>
    >>> x_train = np.array([[1, 2], [3, 4]])
    >>> y_train = np.array([[1], [0]])
    >>>
    >>> optimizer = algorithms.QuasiNewton(
    ...     Input(2) >> Sigmoid(3) >> Sigmoid(1),
    ...     update_function='bfgs'
    ... )
    >>> optimizer.train(x_train, y_train, epochs=10)

    References
    ----------
    [1] Yang Ding, Enkeleida Lushi, Qingguo Li,
        Investigation of quasi-Newton methods for unconstrained optimization.
        http://people.math.sfu.ca/~elushi/project_833.pdf

    [2] Jorge Nocedal, Stephen J. Wright, Numerical Optimization.
        Chapter 6, Quasi-Newton Methods, p. 135-163
    """
    update_function = ChoiceProperty(
        default='bfgs',
        choices={
            'bfgs': bfgs,
            'dfp': dfp,
            'sr1': sr1,
        }
    )
    epsilon = NumberProperty(default=1e-7, minval=0)
    h0_scale = NumberProperty(default=1, minval=0)
    step = WithdrawProperty()

    def init_variables(self):
        with tf.name_scope('quasi-newton'):
            n_parameters = self.network.n_parameters
            self.variables.update(
                inv_hessian=tf.Variable(
                    asfloat(self.h0_scale) * tf.eye(n_parameters),
                    name="inv-hessian",
                    dtype=tf.float32,
                ),
                prev_params=tf.Variable(
                    tf.zeros([n_parameters]),
                    name="prev-params",
                    dtype=tf.float32,
                ),
                prev_full_gradient=tf.Variable(
                    tf.zeros([n_parameters]),
                    name="prev-full-gradient",
                    dtype=tf.float32,
                ),
                iteration=tf.Variable(
                    asfloat(self.last_epoch),
                    name='current-iteration',
                    dtype=tf.float32
                ),
            )

    def init_train_updates(self):
        self.init_variables()

        iteration = self.variables.iteration
        inv_hessian = self.variables.inv_hessian
        prev_params = self.variables.prev_params
        prev_full_gradient = self.variables.prev_full_gradient

        variables = self.network.variables
        params = [var for var in variables.values() if var.trainable]
        param_vector = make_single_vector(params)

        gradients = tf.gradients(self.variables.loss, params)
        full_gradient = make_single_vector(gradients)

        new_inv_hessian = tf.where(
            tf.equal(iteration, 0),
            inv_hessian,
            self.update_function(
                inv_H=inv_hessian,
                delta_w=param_vector - prev_params,
                delta_grad=full_gradient - prev_full_gradient,
                epsilon=self.epsilon
            )
        )
        param_delta = -dot(new_inv_hessian, full_gradient)
        step = self.find_optimal_step(param_vector, param_delta)
        updated_params = param_vector + step * param_delta
        updates = setup_parameter_updates(params, updated_params)

        # We have to compute these values first, otherwise
        # parallelization, in tensorflow, can mix update order
        # and, for example, previous gradient can be equal to
        # current gradient value. It happens because tensorflow
        # try to execute operations in parallel.
        required_variables = [new_inv_hessian, param_vector, full_gradient]
        with tf.control_dependencies(required_variables):
            updates.extend([
                inv_hessian.assign(new_inv_hessian),
                prev_params.assign(param_vector),
                prev_full_gradient.assign(full_gradient),
                iteration.assign(iteration + 1),
            ])

        return updates
