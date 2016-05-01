import theano
import theano.tensor as T
import numpy as np

from neupy.utils import asfloat
from neupy.core.properties import ProperFractionProperty, Property
from .base import MinibatchGradientDescent


__all__ = ('Momentum',)


class Momentum(MinibatchGradientDescent):
    """ Momentum algorithm for :network:`GradientDescent` optimization.

    Parameters
    ----------
    momentum : float
        Control previous gradient ratio. Defaults to ``0.9``.
    nesterov : bool
        Instead of classic momentum computes Nesterov momentum.
        Defaults to ``False``.
    {MinibatchGradientDescent.batch_size}
    {GradientDescent.addons}
    {ConstructableNetwork.connection}
    {ConstructableNetwork.error}
    {BaseNetwork.step}
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
    nesterov = Property(default=False, expected_type=bool)

    def init_layers(self):
        super(Momentum, self).init_layers()
        for layer in self.layers:
            for parameter in layer.parameters:
                parameter_shape = T.shape(parameter).eval()
                parameter.prev_param_delta = theano.shared(
                    name="prev_param_delta_" + parameter.name,
                    value=asfloat(np.zeros(parameter_shape)),
                )

    def init_param_updates(self, layer, parameter):
        step = self.variables.step
        gradient = T.grad(self.variables.error_func, wrt=parameter)

        prev_param_delta = parameter.prev_param_delta
        parameter_delta = self.momentum * prev_param_delta - step * gradient

        if self.nesterov:
            parameter_delta = self.momentum * parameter_delta - step * gradient

        return [
            (parameter, parameter + parameter_delta),
            (prev_param_delta, parameter_delta),
        ]
