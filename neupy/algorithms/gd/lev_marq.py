import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.tensor import slinalg
import numpy as np

from neupy.utils import asfloat
from neupy.core.properties import (BoundedProperty, ChoiceProperty,
                                   WithdrawProperty)
from neupy.algorithms import GradientDescent
from neupy.algorithms.gd import StepSelectionBuiltIn, errors
from neupy.algorithms.utils import parameter_values, setup_parameter_updates


__all__ = ('LevenbergMarquardt',)


def compute_jacobian(errors, parameters):
    """
    Compute jacobian.

    Parameters
    ----------
    errors : Theano variable
        Computed MSE for each sample separetly.

    parameters : list of Theano variable
        Neural network parameters (e.g. weights, biases).

    Returns
    -------
    Theano variable
    """
    n_samples = errors.shape[0]
    J = T.jacobian(errors, wrt=parameters)

    jacobians = []
    for jacobian, parameter in zip(J, parameters):
        jacobian = jacobian.reshape((n_samples, parameter.size))
        jacobians.append(jacobian)

    return T.concatenate(jacobians, axis=1)


class LevenbergMarquardt(StepSelectionBuiltIn, GradientDescent):
    """
    Levenberg-Marquardt algorithm.

    Notes
    -----
    - Network minimizes only Mean Squared Error function.
    - Efficient for small training datasets, because it
      computes gradient per each sample separately.
    - Efficient for small-sized networks.

    Parameters
    ----------
    {GradientDescent.connection}

    mu : float
        Control invertion for J.T * J matrix, defaults to `0.1`.

    mu_update_factor : float
        Factor to decrease the mu if update decrese the error, otherwise
        increse mu by the same factor. Defaults to ``1.2``

    error : {{``mse``}}
        Levenberg-Marquardt works only for quadratic functions.
        Defaults to ``mse``.

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
    >>> lmnet = algorithms.LevenbergMarquardt((2, 3, 1))
    >>> lmnet.train(x_train, y_train)

    See Also
    --------
    :network:`GradientDescent` : GradientDescent algorithm.
    """
    mu = BoundedProperty(default=0.01, minval=0)
    mu_update_factor = BoundedProperty(default=1.2, minval=1)
    error = ChoiceProperty(default='mse', choices={'mse': errors.mse})

    step = WithdrawProperty()

    def init_variables(self):
        super(LevenbergMarquardt, self).init_variables()
        self.variables.update(
            mu=theano.shared(name='lev-marq/mu', value=asfloat(self.mu)),
            last_error=theano.shared(name='lev-marq/last-error', value=np.nan),
        )

    def init_train_updates(self):
        network_output = self.variables.network_output
        prediction_func = self.variables.train_prediction_func
        last_error = self.variables.last_error
        error_func = self.variables.error_func
        mu = self.variables.mu

        new_mu = ifelse(
            T.lt(last_error, error_func),
            mu * self.mu_update_factor,
            mu / self.mu_update_factor,
        )

        se_for_each_sample = (
            (network_output - prediction_func) ** 2
        ).ravel()

        params = parameter_values(self.connection)
        param_vector = T.concatenate([param.flatten() for param in params])

        J = compute_jacobian(se_for_each_sample, params)
        n_params = J.shape[1]

        updated_params = param_vector - slinalg.solve(
            J.T.dot(J) + new_mu * T.eye(n_params),
            J.T.dot(se_for_each_sample)
        )

        updates = [(mu, new_mu)]
        parameter_updates = setup_parameter_updates(params, updated_params)
        updates.extend(parameter_updates)

        return updates

    def on_epoch_start_update(self, epoch):
        super(LevenbergMarquardt, self).on_epoch_start_update(epoch)

        last_error = self.errors.last()
        if last_error is not None:
            self.variables.last_error.set_value(last_error)
