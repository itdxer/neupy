import tensorflow as tf

from neupy.core.properties import BoundedProperty, WithdrawProperty
from neupy.utils import asfloat, flatten, get_variable_size
from neupy.algorithms.gd import StepSelectionBuiltIn
from neupy.algorithms.utils import parameter_values, setup_parameter_updates
from neupy.layers.utils import count_parameters
from .base import GradientDescent


__all__ = ('Hessian',)


def find_hessian_and_gradient(error_function, parameters):
    """
    Compute jacobian.

    Parameters
    ----------
    values : Theano variable
        Computed MSE for each sample separetly.

    parameters : list of Theano variable
        Neural network parameters (e.g. weights, biases).

    Returns
    -------
    Theano variable
    """
    n_parameters = sum(get_variable_size(parameter) for parameter in parameters)
    gradients = tf.gradients(error_function, parameters)
    full_gradient = tf.concat([flatten(grad) for grad in gradients], axis=0)

    full_gradient_shape = tf.shape(full_gradient)
    n_samples = full_gradient_shape[0]

    def compute_gradient_per_value(index, result):
        gradients = tf.gradients(full_gradient[index], parameters)
        hessian = tf.concat(
            [flatten(gradient) for gradient in gradients], axis=0)

        return (index + 1, result.write(index, hessian))

    # import ipdb; ipdb.set_trace()

    _, jacobian = tf.while_loop(
        lambda index, _: index < n_samples,
        compute_gradient_per_value,
        [
            tf.constant(0, tf.int32),
            tf.TensorArray(tf.float32, size=n_samples),
        ]
    )

    return jacobian.stack(), full_gradient


class Hessian(StepSelectionBuiltIn, GradientDescent):
    """
    Hessian gradient decent optimization. This GD algorithm
    variation using second derivative information helps choose better
    gradient direction and as a consequence better weight update
    parameter after each epoch.

    Parameters
    ----------
    penalty_const : float
        Inverse hessian could be singular matrix. For this reason
        algorithm include penalty that add to hessian matrix identity
        multiplied by defined constant. Defaults to ``1``.

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
        param_vector = tf.concat(
            [flatten(param) for param in parameters], axis=0)

        # import ipdb; ipdb.set_trace()
        hessian_matrix, full_gradient = find_hessian_and_gradient(
            self.variables.error_func, parameters
        )

        # import ipdb; ipdb.set_trace()

        solution = tf.matrix_solve(
            hessian_matrix + penalty_const * tf.eye(n_parameters),
            tf.reshape(full_gradient, [-1, 1])
        )
        updated_parameters = param_vector - flatten(solution)
        updates = setup_parameter_updates(parameters, updated_parameters)

        return updates
