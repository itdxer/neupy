import numpy as np
import tensorflow as tf

from neupy.utils import tensorflow_session, flatten, function_name_scope
from neupy.core.properties import (BoundedProperty, ChoiceProperty,
                                   WithdrawProperty)
from neupy.algorithms import BaseGradientDescent
from neupy.algorithms.gd import StepSelectionBuiltIn, errors
from neupy.algorithms.utils import (parameter_values, setup_parameter_updates,
                                    make_single_vector)


__all__ = ('LevenbergMarquardt',)


@function_name_scope
def compute_jacobian(values, parameters):
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
    values_shape = tf.shape(values)
    n_samples = values_shape[0]

    def compute_gradient_per_value(index, result):
        gradients = tf.gradients(values[index], parameters)
        full_gradient = make_single_vector(gradients)
        return (index + 1, result.write(index, full_gradient))

    _, jacobian = tf.while_loop(
        lambda index, _: index < n_samples,
        compute_gradient_per_value,
        [
            tf.constant(0, tf.int32),
            tf.TensorArray(tf.float32, size=n_samples),
        ]
    )

    return jacobian.stack()


class LevenbergMarquardt(StepSelectionBuiltIn, BaseGradientDescent):
    """
    Levenberg-Marquardt algorithm is a variation of the Newton's method.
    It minimizes MSE error. The algorithm approximates Hessian matrix using
    dot product between two jacobian matrices.

    Notes
    -----
    - Method requires all training data during propagation, which means
      it's not allowed to use mini-batches.

    - Network minimizes only Mean Squared Error (MSE) loss function.

    - Efficient for small training datasets, because it
      computes gradient per each sample separately.

    - Efficient for small-sized networks.

    Parameters
    ----------
    {BaseGradientDescent.connection}

    mu : float
        Control invertion for J.T * J matrix, defaults to ``0.1``.

    mu_update_factor : float
        Factor to decrease the mu if update decrese the error, otherwise
        increse mu by the same factor. Defaults to ``1.2``

    error : {{``mse``}}
        Levenberg-Marquardt works only for quadratic functions.
        Defaults to ``mse``.

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

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms
    >>>
    >>> x_train = np.array([[1, 2], [3, 4]])
    >>> y_train = np.array([[1], [0]])
    >>>
    >>> lmnet = algorithms.LevenbergMarquardt((2, 3, 1))
    >>> lmnet.train(x_train, y_train)

    See Also
    --------
    :network:`BaseGradientDescent` : BaseGradientDescent algorithm.
    """
    mu = BoundedProperty(default=0.01, minval=0)
    mu_update_factor = BoundedProperty(default=1.2, minval=1)
    error = ChoiceProperty(default='mse', choices={'mse': errors.mse})

    step = WithdrawProperty()

    def init_variables(self):
        super(LevenbergMarquardt, self).init_variables()
        self.variables.update(
            mu=tf.Variable(self.mu, name='lev-marq/mu'),
            last_error=tf.Variable(np.nan, name='lev-marq/last-error'),
        )

    def init_train_updates(self):
        network_output = self.variables.network_output
        prediction_func = self.variables.train_prediction_func
        last_error = self.variables.last_error
        error_func = self.variables.error_func
        mu = self.variables.mu

        new_mu = tf.where(
            tf.less(last_error, error_func),
            mu * self.mu_update_factor,
            mu / self.mu_update_factor,
        )

        err_for_each_sample = flatten((network_output - prediction_func) ** 2)

        params = parameter_values(self.connection)
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

    def on_epoch_start_update(self, epoch):
        super(LevenbergMarquardt, self).on_epoch_start_update(epoch)

        last_error = self.errors.last()
        if last_error is not None:
            self.variables.last_error.load(last_error, tensorflow_session())
