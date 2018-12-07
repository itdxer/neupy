import tensorflow as tf

from neupy.core.properties import BoundedProperty, WithdrawProperty
from neupy.utils import asfloat, flatten, function_name_scope
from neupy.layers.utils import count_parameters
from neupy.algorithms.gd import StepSelectionBuiltIn
from neupy.algorithms.utils import (parameter_values, setup_parameter_updates,
                                    make_single_vector)
from .base import BaseGradientDescent


__all__ = ('Hessian',)


@function_name_scope
def find_hessian_and_gradient(error_function, parameters):
    """
    Compute jacobian.

    Parameters
    ----------
    values : Tensorfow variable
        Computed MSE for each sample separetly.

    parameters : list of Tensorfow variable
        Neural network parameters (e.g. weights, biases).

    Returns
    -------
    Tensorfow variable
    """
    gradients = tf.gradients(error_function, parameters)
    full_gradient = make_single_vector(gradients)

    full_gradient_shape = tf.shape(full_gradient)
    n_samples = full_gradient_shape[0]

    def compute_gradient_per_value(index, result):
        gradients = tf.gradients(full_gradient[index], parameters)
        hessian = make_single_vector(gradients)
        return (index + 1, result.write(index, hessian))

    _, hessian = tf.while_loop(
        lambda index, _: index < n_samples,
        compute_gradient_per_value,
        [
            tf.constant(0, tf.int32),
            tf.TensorArray(tf.float32, size=n_samples),
        ]
    )

    return hessian.stack(), full_gradient


class Hessian(StepSelectionBuiltIn, BaseGradientDescent):
    """
    Hessian gradient decent optimization, also known as Newton's method. This
    algorithm uses second-order derivative (hessian matrix) in order to
    choose correct step during the training itration. Because of this,
    method doesn't have ``step`` parameter.

    Parameters
    ----------
    penalty_const : float
        Inverse hessian could be singular matrix. For this reason
        algorithm include penalty that add to hessian matrix identity
        multiplied by defined constant. Defaults to ``1``.

    {BaseGradientDescent.connection}

    {BaseGradientDescent.error}

    {BaseGradientDescent.show_epoch}

    {BaseGradientDescent.shuffle_data}

    {BaseGradientDescent.epoch_end_signal}

    {BaseGradientDescent.train_end_signal}

    {BaseGradientDescent.verbose}

    {BaseGradientDescent.addons}

    Attributes
    ----------
    {BaseGradientDescent.Attributes}

    Methods
    -------
    {BaseGradientDescent.Methods}

    Notes
    -----
    - Method requires all training data during propagation, which means
      it's not allowed to use mini-batches.

    - This method calculates full hessian matrix which means it will compute
      matrix with NxN parameters, where N = number of parameters in the
      network.

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms
    >>>
    >>> x_train = np.array([[1, 2], [3, 4]])
    >>> y_train = np.array([[1], [0]])
    >>>
    >>> mnet = algorithms.Hessian((2, 3, 1))
    >>> mnet.train(x_train, y_train)

    See Also
    --------
    :network:`HessianDiagonal` : Hessian diagonal approximation.
    """
    penalty_const = BoundedProperty(default=1, minval=0)

    step = WithdrawProperty()

    def init_train_updates(self):
        penalty_const = asfloat(self.penalty_const)

        n_parameters = count_parameters(self.connection)
        parameters = parameter_values(self.connection)
        param_vector = make_single_vector(parameters)

        hessian_matrix, full_gradient = find_hessian_and_gradient(
            self.variables.error_func, parameters
        )
        parameter_update = tf.matrix_solve(
            hessian_matrix + penalty_const * tf.eye(n_parameters),
            tf.reshape(full_gradient, [-1, 1])
        )
        updated_parameters = param_vector - flatten(parameter_update)
        updates = setup_parameter_updates(parameters, updated_parameters)

        return updates
