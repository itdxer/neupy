import numpy as np
import tensorflow as tf

from neupy.utils import (
    tensorflow_session, flatten,
    function_name_scope, make_single_vector,
)
from neupy.core.properties import (BoundedProperty, ChoiceProperty,
                                   WithdrawProperty)
from neupy.algorithms import BaseOptimizer
from neupy.algorithms.gd import objectives
from neupy.utils.tf_utils import setup_parameter_updates


__all__ = ('LevenbergMarquardt',)


@function_name_scope
def compute_jacobian(values, parameters):
    """
    Compute Jacobian matrix.

    Parameters
    ----------
    values : Tensorfow variable
        Computed MSE for each sample separately.

    parameters : list of Tensorfow variable
        Neural network parameters (e.g. weights, biases).

    Returns
    -------
    Tensorfow variable
    """
    values_shape = tf.shape(values)
    n_samples = values_shape[0]

    def compute_gradient_per_value(index, result):
        gradients = tf.gradients(values[index], parameters)
        full_gradient = make_single_vector(gradients)
        return index + 1, result.write(index, full_gradient)

    _, jacobian = tf.while_loop(
        lambda index, _: index < n_samples,
        compute_gradient_per_value,
        [
            tf.constant(0, tf.int32),
            tf.TensorArray(tf.float32, size=n_samples),
        ]
    )

    return jacobian.stack()


class LevenbergMarquardt(BaseOptimizer):
    """
    Levenberg-Marquardt algorithm is a variation of the Newton's method.
    It minimizes MSE error. The algorithm approximates Hessian matrix using
    dot product between two jacobian matrices.

    Notes
    -----
    - Method requires all training data during propagation, which means
      it cannot be trained with mini-batches.

    - Network minimizes only Mean Squared Error (MSE) loss function.

    - Efficient for small training datasets, because it
      computes gradient per each sample separately.

    - Efficient for small-sized networks.

    Parameters
    ----------
    {BaseOptimizer.network}

    mu : float
        Control inversion for J.T * J matrix, defaults to ``0.1``.

    mu_update_factor : float
        Factor to decrease the mu if error was reduced after last update,
        otherwise increase mu by the same factor. Defaults to ``1.2``

    error : {{``mse``}}
        Levenberg-Marquardt works only for quadratic functions.
        Defaults to ``mse``.

    {BaseOptimizer.show_epoch}

    {BaseOptimizer.shuffle_data}

    {BaseOptimizer.signals}

    {BaseOptimizer.verbose}

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
    >>> network = Input(2) >> Sigmoid(3) >> Sigmoid(1)
    >>> optimizer = algorithms.LevenbergMarquardt(network)
    >>> optimizer.train(x_train, y_train)

    See Also
    --------
    :network:`BaseOptimizer` : BaseOptimizer algorithm.
    """
    mu = BoundedProperty(default=0.01, minval=0)
    mu_update_factor = BoundedProperty(default=1.2, minval=1)
    loss = ChoiceProperty(default='mse', choices={'mse': objectives.mse})

    step = WithdrawProperty()
    regularizer = WithdrawProperty()

    def init_functions(self):
        self.variables.update(
            mu=tf.Variable(self.mu, name='lev-marq/mu'),
            last_error=tf.Variable(np.nan, name='lev-marq/last-error'),
        )
        super(LevenbergMarquardt, self).init_functions()

    def init_train_updates(self):
        training_outputs = self.network.training_outputs
        last_error = self.variables.last_error
        error_func = self.variables.loss
        mu = self.variables.mu

        new_mu = tf.where(
            tf.less(last_error, error_func),
            mu * self.mu_update_factor,
            mu / self.mu_update_factor,
        )

        err_for_each_sample = flatten((self.target - training_outputs) ** 2)

        variables = self.network.variables
        params = [var for var in variables.values() if var.trainable]
        param_vector = make_single_vector(params)

        J = compute_jacobian(err_for_each_sample, params)
        J_T = tf.transpose(J)
        n_params = J.shape[1]

        parameter_update = tf.matrix_solve(
            tf.matmul(J_T, J) + new_mu * tf.eye(n_params.value),
            tf.matmul(J_T, tf.expand_dims(err_for_each_sample, 1))
        )
        updated_params = param_vector - flatten(parameter_update)

        updates = [(mu, new_mu)]
        parameter_updates = setup_parameter_updates(params, updated_params)
        updates.extend(parameter_updates)

        return updates

    def one_training_update(self, X_train, y_train):
        if self.errors.train:
            last_error = self.errors.train[-1]
            self.variables.last_error.load(last_error, tensorflow_session())

        return super(LevenbergMarquardt, self).one_training_update(
            X_train, y_train)
