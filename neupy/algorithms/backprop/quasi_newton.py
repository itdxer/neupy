from numpy import eye, newaxis, sign, isinf, clip, inner, outer
from numpy.linalg import norm

from neupy.core.properties import (ChoiceProperty, NonNegativeNumberProperty,
                                   BetweenZeroAndOneProperty)
from neupy.algorithms.utils import (matrix_list_in_one_vector,
                                    vector_to_list_of_matrix)
from ..steps.wolfe_search import WolfeSearch
from .backpropagation import Backpropagation


__all__ = ('QuasiNewton',)


def bfgs(inverse_hessian, weight_delta, gradient_delta, maxrho=1e4):
    ident_matrix = eye(inverse_hessian.shape[0], dtype=int)

    rho = (1. / gradient_delta.dot(weight_delta))

    if isinf(rho):
        rho = maxrho * sign(rho)

    param1 = ident_matrix - outer(weight_delta, gradient_delta) * rho
    param2 = ident_matrix - outer(gradient_delta, weight_delta) * rho
    param3 = rho * outer(weight_delta, weight_delta)

    return param1.dot(inverse_hessian).dot(param2) + param3


def dfp(inverse_hessian, weight_delta, gradient_delta, maxnum=1e5):
    gradient_delta_t = gradient_delta[newaxis, :]
    gradient_delta = gradient_delta[:, newaxis]
    weight_delta = weight_delta[:, newaxis]

    quasi_dot_gradient = inverse_hessian.dot(gradient_delta)

    param1 = (
        weight_delta.dot(weight_delta.T)
    ) / (
        gradient_delta_t.dot(weight_delta)
    )
    param2_numerator = clip(
        quasi_dot_gradient.dot(gradient_delta_t) * inverse_hessian,
        a_min=-maxnum,
        a_max=maxnum
    )
    param2_denominator = gradient_delta_t.dot(quasi_dot_gradient)
    param2 = param2_numerator / param2_denominator

    return inverse_hessian + param1 - param2


def psb(inverse_hessian, weight_delta, gradient_delta, **options):
    gradient_delta_t = gradient_delta.T
    param = weight_delta - inverse_hessian.dot(gradient_delta)

    devider = (1. / inner(gradient_delta, gradient_delta)).item(0)
    param1 = outer(param, gradient_delta) + outer(gradient_delta, param)
    param2 = (
        inner(gradient_delta, param)
    ).item(0) * outer(gradient_delta, gradient_delta_t)

    return inverse_hessian + param1 * devider - param2 * devider ** 2


def sr1(inverse_hessian, weight_delta, gradient_delta, epsilon=1e-8):
    param = weight_delta - inverse_hessian.dot(gradient_delta)
    denominator = inner(param, gradient_delta)

    if abs(denominator) < epsilon * norm(param) * norm(gradient_delta):
        return inverse_hessian

    return inverse_hessian + outer(param, param) / denominator


class QuasiNewton(Backpropagation):
    """ Quasi-Newton :network:`Backpropagation` algorithm optimization.

    Parameters
    ----------
    update_function : {{'bfgs', 'dfp', 'psb', 'sr1'}}
        Update function. Defaults to ``bfgs``.
    h0_scale : float
        Factor that scale indentity matrix H0 on the first
        iteration step. Defaults to ``1``.
    gradient_tol : float
        In the gradient less than this value algorithm will stop training
        procedure. Defaults to ``1e-5``.
    {optimizations}
    {raw_predict_param}
    {full_params}

    Methods
    -------
    {supervised_train}
    {full_methods}

    Examples
    --------
    Simple example

    >>> import numpy as np
    >>> from neupy import algorithms
    >>>
    >>> x_train = np.array([[1, 2], [3, 4]])
    >>> y_train = np.array([[1], [0]])
    >>>
    >>> qnnet = algorithms.QuasiNewton(
    ...     (2, 3, 1),
    ...     update_function='bfgs',
    ...     verbose=False
    ... )
    >>> qnnet.train(x_train, y_train)

    See Also
    --------
    :network:`Backpropagation` : Backpropagation algorithm.
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
    h0_scale = NonNegativeNumberProperty(default=1)
    gradient_tol = BetweenZeroAndOneProperty(default=1e-5)

    default_optimizations = [WolfeSearch]

    def get_weight_delta(self, output_train, target_train):
        gradients = self.get_gradient(output_train, target_train)
        gradient = matrix_list_in_one_vector(gradients)

        if norm(gradient) < self.gradient_tol:
            raise StopIteration("Gradient norm less than {}"
                                "".format(self.gradient_tol))

        train_layers = self.train_layers
        weight = matrix_list_in_one_vector(
            (layer.weight for layer in train_layers)
        )

        if hasattr(self, 'prev_gradient'):
            # In first epoch we didn't have previous weights and
            # gradients. For this reason we skip quasi coefitient
            # computation.
            inverse_hessian = self.update_function(
                self.prev_inverse_hessian,
                weight - self.prev_weight,
                gradient - self.prev_gradient
            )
        else:
            inverse_hessian = self.h0_scale * eye(weight.size, dtype=int)

        self.prev_weight = weight.copy()
        self.prev_gradient = gradient.copy()
        self.prev_inverse_hessian = inverse_hessian

        return vector_to_list_of_matrix(
            -inverse_hessian.dot(gradient),
            (layer.size for layer in train_layers)
        )
