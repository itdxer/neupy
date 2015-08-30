from numpy import eye, newaxis, sign
from numpy.linalg import norm

from neuralpy.core.properties import (ChoiceProperty,
                                      NonNegativeNumberProperty,
                                      BetweenZeroAndOneProperty)
from neuralpy.algorithms.utils import (matrix_list_in_one_vector,
                                       vector_to_list_of_matrix)
from .steps.wolfe_search import WolfeSearch
from .backpropagation import Backpropagation


__all__ = ('QuasiNewton',)


def bfgs(quasi_update, weight_delta, gradient_delta, maxrho=1e4):
    ident_matrix = eye(quasi_update.shape[0], dtype=int)

    rho = (1. / gradient_delta.dot(weight_delta))

    if abs(rho) > maxrho:
        rho = maxrho * sign(rho)

    # print(rho)
    gradient_delta_t = gradient_delta[newaxis, :]
    gradient_delta = gradient_delta[:, newaxis]
    weight_delta_t = weight_delta[newaxis, :]
    weight_delta = weight_delta[:, newaxis]

    param1 = ident_matrix - weight_delta * gradient_delta_t * rho
    param2 = ident_matrix - gradient_delta * weight_delta_t * rho
    param3 = rho * weight_delta * weight_delta_t

    return param1.dot(quasi_update).dot(param2) + param3


def dfp(quasi_update, weight_delta, gradient_delta, **options):
    gradient_delta_t = gradient_delta.T
    quasi_dot_gradient = quasi_update.dot(gradient_delta)

    param1 = (
        weight_delta.dot(weight_delta.T)
    ) / (
        gradient_delta_t.dot(weight_delta)
    )
    param2 = (
        quasi_dot_gradient.dot(gradient_delta_t).dot(quasi_update)
    ) / (
        gradient_delta_t.dot(quasi_dot_gradient)
    )

    return quasi_update + param1 - param2


def psb(quasi_update, weight_delta, gradient_delta, **options):
    gradient_delta_t = gradient_delta.T
    param = weight_delta - quasi_update * gradient_delta

    devider = (1. / (gradient_delta_t * gradient_delta)).item(0)
    param1 = param * gradient_delta_t + gradient_delta * param.T
    param2 = (
        gradient_delta_t * param
    ).item(0) * gradient_delta * gradient_delta_t

    return quasi_update + param1 * devider - param2 * devider ** 2


def sr1(quasi_update, weight_delta, gradient_delta, epsilon=1e-8):
    param = weight_delta - quasi_update.dot(gradient_delta)
    param_t = param.T
    denominator = param_t.dot(gradient_delta)

    if abs(denominator) < epsilon * norm(param) * norm(gradient_delta):
        return quasi_update

    return quasi_update + param.dot(param_t) / denominator


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
    >>> from neuralpy import algorithms
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
            quasi_update = self.update_function(
                self.prev_quasi_update,
                weight - self.prev_weight,
                gradient - self.prev_gradient
            )
        else:
            quasi_update = self.h0_scale * eye(weight.size, dtype=int)

        self.prev_weight = weight.copy()
        self.prev_gradient = gradient.copy()
        self.prev_quasi_update = quasi_update

        return vector_to_list_of_matrix(
            -quasi_update.dot(gradient),
            (layer.size for layer in train_layers)
        )
