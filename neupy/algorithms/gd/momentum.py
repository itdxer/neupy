import theano
import theano.tensor as T
import numpy as np

from neupy.utils import asfloat
from neupy.core.properties import ProperFractionProperty
from .base import MinibatchGradientDescent


__all__ = ('Momentum',)


class Momentum(MinibatchGradientDescent):
    """ Momentum algorithm for :network:`GradientDescent` optimization.

    Parameters
    ----------
    momentum : float
        Control previous gradient ratio. Defaults to ``0.9``.
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
    >>> mnet = algorithms.Momentum(
    ...     (2, 3, 1),
    ...     verbose=False
    ... )
    >>> mnet.train(x_train, y_train)

    See Also
    --------
    :network:`GradientDescent` : GradientDescent algorithm.
    """
    momentum = ProperFractionProperty(default=0.9)

    def init_layers(self):
        super(Momentum, self).init_layers()
        for layer in self.train_layers:
            for parameter in layer.parameters:
                parameter_shape = T.shape(parameter).eval()
                parameter.prev_gradient = theano.shared(
                    name="prev_grad_" + parameter.name,
                    value=asfloat(np.zeros(parameter_shape)),
                )

    def init_param_updates(self, layer, parameter):
        step = layer.step or self.variables.step
        gradient = T.grad(self.variables.error_func, wrt=parameter)

        prev_gradient = parameter.prev_gradient
        parameter_delta = -self.momentum * prev_gradient - step * gradient

        return [
            (parameter, parameter + parameter_delta),
            (prev_gradient, gradient),
        ]
