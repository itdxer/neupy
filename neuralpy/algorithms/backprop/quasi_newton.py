from operator import mul

from numpy import identity, asmatrix

from neuralpy.core.properties import ChoiceProperty
from neuralpy.algorithms.utils import (matrix_list_in_one_vector,
                                       vector_to_list_of_matrix)
from .backpropagation import Backpropagation


__all__ = ('QuasiNewton',)


def bfgs(quasi_update, weight_delta, gradient_delta):
    ident_matrix = identity(quasi_update.shape[0])
    gradient_delta_t = gradient_delta.T
    weight_delta_t = weight_delta.T

    coef = (1. / gradient_delta_t.dot(weight_delta)).item(0)

    param1 = ident_matrix - weight_delta.dot(gradient_delta_t).dot(coef)
    param2 = ident_matrix - gradient_delta.dot(weight_delta_t).dot(coef)
    param3 = weight_delta.dot(weight_delta_t).dot(coef)

    return param1.T.dot(quasi_update).dot(param2) + param3


def dfp(quasi_update, weight_delta, gradient_delta):
    gradient_delta_t = gradient_delta.T
    quasi_dot_gradient = quasi_update * gradient_delta

    param1 = (
        weight_delta * weight_delta.T
    ) / (
        gradient_delta_t * weight_delta
    )
    param2 = (
        quasi_dot_gradient * gradient_delta_t * quasi_update
    ) / (
        gradient_delta_t * quasi_dot_gradient
    )

    return quasi_update + param1 - param2


def psb(quasi_update, weight_delta, gradient_delta):
    gradient_delta_t = gradient_delta.T
    param = weight_delta - quasi_update * gradient_delta

    devider = (1. / (gradient_delta_t * gradient_delta)).item(0)
    param1 = param * gradient_delta_t + gradient_delta * param.T
    param2 = (
        gradient_delta_t * param
    ).item(0) * gradient_delta * gradient_delta_t

    return quasi_update + param1 * devider - param2 * devider ** 2


def sr1(quasi_update, weight_delta, gradient_delta):
    param = weight_delta - quasi_update.dot(gradient_delta)
    param_t = param.T
    return quasi_update + (param * param_t) / (param_t.dot(gradient_delta))


class QuasiNewton(Backpropagation):
    """ Quasi-Newton :network:`Backpropagation` algorithm optimization.

    Parameters
    ----------
    update_function : {{'bfgs', 'dfp', 'psb', 'sr1'}}
        Update function. Defaults to ``bfgs``.
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

    def learn(self, output_train, target_train):
        weight_deltas = super(QuasiNewton, self).learn(output_train,
                                                       target_train)
        train_layers = self.train_layers

        weight = matrix_list_in_one_vector(
            (layer.weight for layer in train_layers)
        )
        gradient = matrix_list_in_one_vector(weight_deltas)

        if hasattr(self, 'prev_gradient'):
            # In first epoch we didn't have previous weights and
            # gradients. For this reason we skip quasi coefitient
            # computation.
            weight_delta = asmatrix(weight - self.prev_weight).T
            gradient_delta = asmatrix(gradient - self.prev_gradient).T

            quasi_update = self.update_function(self.prev_quasi_update,
                                                weight_delta, gradient_delta)
        else:
            update_vector_size = sum(
                mul(*layer.size) for layer in train_layers
            )
            quasi_update = identity(update_vector_size)

        self.prev_weight = weight.copy()
        self.prev_gradient = gradient.copy()
        self.prev_quasi_update = quasi_update

        return vector_to_list_of_matrix(
            -quasi_update.dot(gradient),
            (layer.size for layer in train_layers)
        )
