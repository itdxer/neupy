import theano
import theano.typed_list
import theano.tensor as T

from neupy.core.properties import BoundedProperty
from neupy.utils import asfloat
from neupy.algorithms.gd import NoStepSelection
from neupy.algorithms.utils import (parameters2vector, count_parameters,
                                    iter_parameters, setup_parameter_updates)
from .base import GradientDescent


__all__ = ('Hessian',)


def find_hessian_and_gradient(error_function, parameters):
    """ Find Hessian and gradient for the Neural Network cost function.

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
        second_derivatives = []
        g = full_gradient[i]
        for parameter in parameters:
            second_derivative = T.grad(g, wrt=parameter)
            second_derivatives.append(second_derivative.flatten())

        return T.concatenate(second_derivatives)

    hessian, _ = theano.scan(
        find_hessian,
        sequences=T.arange(n_parameters),
        non_sequences=[full_gradient] + parameters,
    )
    hessian_matrix = hessian.reshape((n_parameters, n_parameters))

    return hessian_matrix, full_gradient


class Hessian(NoStepSelection, GradientDescent):
    """ Hessian gradient decent optimization. This GD algorithm
    variation using second derivative information helps choose better
    gradient direction and as a consequence better weight update
    parameter after eqch epoch.

    Parameters
    ----------
    penalty_const : float
        Inverse hessian could be singular matrix. For this reason
        algorithm include penalty that add to hessian matrix identity
        multiplied by defined constant. Defaults to ``1``.
    {GradientDescent.addons}
    {ConstructableNetwork.connection}
    {ConstructableNetwork.error}
    {BaseNetwork.show_epoch}
    {BaseNetwork.shuffle_data}
    {BaseNetwork.epoch_end_signal}
    {BaseNetwork.train_end_signal}
    {Verbose.verbose}

    Methods
    -------
    {BaseSkeleton.predict}
    {SupervisedLearning.train}
    {BaseSkeleton.fit}

    See Also
    --------
    :network:`HessianDiagonal` : Hessian diagonal approximation.
    """
    penalty_const = BoundedProperty(default=1, minval=0)

    def init_train_updates(self):
        n_parameters = count_parameters(self)
        parameters = list(iter_parameters(self))
        param_vector = parameters2vector(self)
        penalty_const = asfloat(self.penalty_const)

        hessian_matrix, full_gradient = find_hessian_and_gradient(
            self.variables.error_func, parameters
        )
        hessian_inverse = T.nlinalg.matrix_inverse(
            hessian_matrix + penalty_const * T.eye(n_parameters)
        )

        updated_parameters = param_vector - hessian_inverse.dot(full_gradient)
        updates = setup_parameter_updates(parameters, updated_parameters)

        return updates
