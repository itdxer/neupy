import theano
import theano.typed_list
import theano.tensor as T
from theano.tensor import slinalg

from neupy.core.properties import BoundedProperty, WithdrawProperty
from neupy.utils import asfloat
from neupy.algorithms.gd import StepSelectionBuiltIn
from neupy.algorithms.utils import parameter_values, setup_parameter_updates
from neupy.layers.utils import count_parameters
from .base import GradientDescent


__all__ = ('Hessian',)


def find_hessian_and_gradient(error_function, parameters):
    """
    Find Hessian and gradient for the Neural Network cost function.

    Parameters
    ----------
    function : Theano function

    parameters : list
        List of all Neural Network parameters.

    Returns
    -------
    Theano function
    """
    n_parameters = T.sum([parameter.size for parameter in parameters])
    gradients = T.grad(error_function, wrt=parameters)
    full_gradient = T.concatenate([grad.flatten() for grad in gradients])

    def find_hessian(i, full_gradient, *parameters):
        second_derivatives = T.grad(full_gradient[i], wrt=parameters)
        return T.concatenate([s.flatten() for s in second_derivatives])

    hessian, _ = theano.scan(
        find_hessian,
        sequences=T.arange(n_parameters),
        non_sequences=[full_gradient] + parameters,
    )
    hessian_matrix = hessian.reshape((n_parameters, n_parameters))

    return hessian_matrix, full_gradient


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
        n_parameters = count_parameters(self.connection)
        parameters = parameter_values(self.connection)
        param_vector = T.concatenate([param.flatten() for param in parameters])
        penalty_const = asfloat(self.penalty_const)

        hessian_matrix, full_gradient = find_hessian_and_gradient(
            self.variables.error_func, parameters
        )

        updated_parameters = param_vector - slinalg.solve(
            hessian_matrix + penalty_const * T.eye(n_parameters),
            full_gradient
        )
        updates = setup_parameter_updates(parameters, updated_parameters)

        return updates
