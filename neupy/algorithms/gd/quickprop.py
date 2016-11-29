from __future__ import division

import theano
import theano.tensor as T
from theano.ifelse import ifelse
import numpy as np

from neupy.core.properties import BoundedProperty
from neupy.utils import asfloat
from .base import GradientDescent


__all__ = ('Quickprop',)


class Quickprop(GradientDescent):
    """
    Quickprop :network:`GradientDescent` algorithm optimization.

    Parameters
    ----------
    upper_bound : float
        Maximum possible value for weight update.
        Defaults to ``1``.

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
    >>> qpnet = algorithms.Quickprop((2, 3, 1))
    >>> qpnet.train(x_train, y_train)

    See Also
    --------
    :network:`GradientDescent` : GradientDescent algorithm.
    """
    upper_bound = BoundedProperty(default=1, minval=0)

    def init_param_updates(self, layer, parameter):
        step = self.variables.step

        parameter_shape = T.shape(parameter).eval()
        prev_delta = theano.shared(
            name="{}/prev-delta".format(parameter.name),
            value=asfloat(np.zeros(parameter_shape)),
        )
        prev_gradient = theano.shared(
            name="{}/prev-grad".format(parameter.name),
            value=asfloat(np.zeros(parameter_shape)),
        )

        gradient = T.grad(self.variables.error_func, wrt=parameter)
        grad_delta = T.abs_(prev_gradient - gradient)

        parameter_delta = ifelse(
            T.eq(self.variables.epoch, 1),
            gradient,
            T.clip(
                T.abs_(prev_delta) * gradient / grad_delta,
                -self.upper_bound,
                self.upper_bound
            )
        )
        return [
            (parameter, parameter - step * parameter_delta),
            (prev_gradient, gradient),
            (prev_delta, parameter_delta),
        ]
