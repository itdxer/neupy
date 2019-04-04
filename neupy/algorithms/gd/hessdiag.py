from __future__ import division

import tensorflow as tf

from neupy.core.properties import ProperFractionProperty
from neupy.utils import flatten, make_single_vector
from neupy.utils.tf_utils import setup_parameter_updates

from .base import BaseOptimizer


__all__ = ('HessianDiagonal',)


class HessianDiagonal(BaseOptimizer):
    """
    Algorithm that uses calculates only diagonal values from the Hessian matrix
    and uses it instead of the Hessian matrix.

    Parameters
    ----------
    min_eigval : float
        Set up minimum eigenvalue for Hessian diagonal matrix. After a few
        iteration elements will be extremely small and matrix inverse
        produce huge number in hessian diagonal elements. This
        parameter control diagonal elements size. Defaults to ``1e-2``.

    {BaseOptimizer.Parameters}

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
    >>> optimizer = algorithms.HessianDiagonal(network)
    >>> optimizer.train(x_train, y_train)

    Notes
    -----
    - Method requires all training data during propagation, which means
      it cannot be trained with mini-batches.

    See Also
    --------
    :network:`BaseOptimizer` : BaseOptimizer algorithm.
    :network:`Hessian` : Newton's method.
    """
    min_eigval = ProperFractionProperty(default=1e-2)

    def init_train_updates(self):
        step = self.step
        inv_min_eigval = 1 / self.min_eigval
        variables = self.network.variables
        parameters = [var for var in variables.values() if var.trainable]
        param_vector = make_single_vector(parameters)

        gradients = tf.gradients(self.variables.loss, parameters)
        full_gradient = make_single_vector(gradients)

        second_derivatives = []
        for parameter, gradient in zip(parameters, gradients):
            second_derivative, = tf.gradients(gradient, parameter)
            second_derivatives.append(flatten(second_derivative))

        hessian_diag = tf.concat(second_derivatives, axis=0)

        # it's easier to clip inverse hessian rather than the hessian,.
        inv_hessian_diag = tf.clip_by_value(
            # inverse for diagonal matrix easy to compute with
            # elementwise inverse operation.
            1 / hessian_diag,
            -inv_min_eigval,
            inv_min_eigval,
        )
        updates = setup_parameter_updates(
            parameters,
            param_vector - step * full_gradient * inv_hessian_diag
        )
        return updates
