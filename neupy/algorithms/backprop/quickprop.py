from __future__ import division

import copy

import theano
import theano.tensor as T
import numpy as np
from numpy import where, sign, abs as np_abs
from numpy.linalg import norm

from neupy.core.properties import (NonNegativeNumberProperty,
                                   BetweenZeroAndOneProperty)
from neupy.network import StopNetworkTraining
from neupy.utils import asfloat
from .backpropagation import Backpropagation


__all__ = ('Quickprop',)


class Quickprop(Backpropagation):
    """ Quickprop :network:`Backpropagation` algorithm optimization.

    Parameters
    ----------
    upper_bound : float
        Maximum possible value for weight update. Defaults to ``1``.
    {optimizations}
    {full_params}

    Methods
    -------
    {supervised_train}
    {predict_raw}
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
    >>> qpnet = algorithms.Quickprop(
    ...     (2, 3, 1),
    ...     verbose=False
    ... )
    >>> qpnet.train(x_train, y_train)

    See Also
    --------
    :network:`Backpropagation` : Backpropagation algorithm.
    """
    upper_bound = NonNegativeNumberProperty(default=1)

    def init_layers(self):
        super(Quickprop, self).init_layers()
        for layer in self.train_layers:
            for parameter in layer.parameters:
                parameter_shape = T.shape(parameter).eval()
                parameter.prev_delta = theano.shared(
                    name="prev_delta_" + parameter.name,
                    value=asfloat(-2 * np.ones(parameter_shape)),
                )
                parameter.prev_gradient = theano.shared(
                    name="prev_grad_" + parameter.name,
                    value=asfloat(np.zeros(parameter_shape)),
                )

    def init_layer_param_updates(self, layer, parameter):
        step = layer.step or self.variables.step
        gradient = T.grad(self.variables.error_func, wrt=parameter)

        prev_delta = parameter.prev_delta
        prev_gradient = parameter.prev_gradient

        parameter_delta = T.clip(
            T.abs_(prev_delta) * gradient / T.abs_(prev_gradient - gradient),
            -self.upper_bound,
            self.upper_bound
        )
        return [
            (parameter, parameter - step * parameter_delta),
            (prev_gradient, gradient),
            (prev_delta, parameter_delta),
        ]
