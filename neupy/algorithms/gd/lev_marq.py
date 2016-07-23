import theano
import theano.tensor as T
from theano.ifelse import ifelse
import numpy as np

from neupy.utils import asfloat
from neupy.network import errors
from neupy.core.properties import BoundedProperty, ChoiceProperty
from neupy.algorithms import GradientDescent
from neupy.algorithms.gd import NoStepSelection
from neupy.algorithms.utils import (parameters2vector, iter_parameters,
                                    setup_parameter_updates)


__all__ = ('LevenbergMarquardt',)


def compute_jaccobian(errors, parameters):
    """
    Compute Jacobbian.

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

    def find_jacobbian(i, errors, *params):
        return T.grad(T.sum(errors[i]), wrt=params)

    J, _ = theano.scan(
        find_jacobbian,
        sequences=T.arange(n_samples),
        non_sequences=[errors] + parameters
    )

    jaccobians = []
    for jaccobian, parameter in zip(J, parameters):
        jaccobian = jaccobian.reshape((n_samples, parameter.size))
        jaccobians.append(jaccobian)

    return T.concatenate(jaccobians, axis=1)


class LevenbergMarquardt(NoStepSelection, GradientDescent):
    """
    Levenberg-Marquardt algorithm.

    Notes
    -----
    * Network minimizes only Mean Squared Error function.

    Parameters
    ----------
    mu : float
        Control invertion for J.T * J matrix, defaults to `0.1`.
    mu_update_factor : float
        Factor to decrease the mu if update decrese the error, otherwise
        increse mu by the same factor. Defaults to ``1.2``
    error: {{'mse'}}
        Levenberg-Marquardt works only for quadratic functions.
        Defaults to ``mse``.
    {GradientDescent.Parameters}

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
    >>> lmnet = algorithms.LevenbergMarquardt(
    ...     (2, 3, 1),
    ...     verbose=False
    ... )
    >>> lmnet.train(x_train, y_train)

    See Also
    --------
    :network:`GradientDescent` : GradientDescent algorithm.
    """

    mu = BoundedProperty(default=0.01, minval=0)
    mu_update_factor = BoundedProperty(default=1.2, minval=1)
    error = ChoiceProperty(default='mse', choices={'mse': errors.mse})

    def init_variables(self):
        super(LevenbergMarquardt, self).init_variables()
        self.variables.update(
            mu=theano.shared(name='mu', value=asfloat(self.mu)),
            last_error=theano.shared(name='last_error', value=np.nan),
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

        mse_for_each_sample = T.mean(
            (network_output - prediction_func) ** 2,
            axis=1
        )

        params = list(iter_parameters(self))
        param_vector = parameters2vector(self)

        J = compute_jaccobian(mse_for_each_sample, params)
        n_params = J.shape[1]

        updated_params = param_vector - T.nlinalg.matrix_inverse(
            J.T.dot(J) + new_mu * T.eye(n_params)
        ).dot(J.T).dot(mse_for_each_sample)

        updates = [(mu, new_mu)]
        parameter_updates = setup_parameter_updates(params, updated_params)
        updates.extend(parameter_updates)

        return updates

    def on_epoch_start_update(self, epoch):
        super(LevenbergMarquardt, self).on_epoch_start_update(epoch)

        last_error = self.errors.last()
        if last_error is not None:
            self.variables.last_error.set_value(last_error)
