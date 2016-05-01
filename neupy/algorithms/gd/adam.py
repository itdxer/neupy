import theano
import theano.tensor as T
import numpy as np

from neupy.utils import asfloat
from neupy.core.properties import ProperFractionProperty, NumberProperty
from .base import MinibatchGradientDescent


__all__ = ('Adam',)


class Adam(MinibatchGradientDescent):
    """ Adam algorithm.

    Parameters
    ----------
    beta1 : float
        Decay rate. Value need to be between ``0`` and ``1``.
        Defaults to ``0.95``.
    beta2 : float
        Decay rate. Value need to be between ``0`` and ``1``.
        Defaults to ``0.95``.
    epsilon : float
        Value need to be greater than ``0``. Defaults to ``1e-5``.
    step : float
        Learning rate, defaults to ``0.001``.
    {MinibatchGradientDescent.batch_size}
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
    """
    step = NumberProperty(default=0.001, minval=0)
    beta1 = ProperFractionProperty(default=0.9)
    beta2 = ProperFractionProperty(default=0.999)
    epsilon = NumberProperty(default=1e-7, minval=0)

    def init_layers(self):
        super(Adam, self).init_layers()
        for layer in self.layers:
            for parameter in layer.parameters:
                parameter_shape = T.shape(parameter).eval()
                parameter.prev_first_moment = theano.shared(
                    name="prev_first_moment_" + parameter.name,
                    value=asfloat(np.zeros(parameter_shape)),
                )
                parameter.prev_second_moment = theano.shared(
                    name="prev_second_moment_" + parameter.name,
                    value=asfloat(np.zeros(parameter_shape)),
                )

    def init_param_updates(self, layer, parameter):
        epoch = self.variables.epoch
        prev_first_moment = parameter.prev_first_moment
        prev_second_moment = parameter.prev_second_moment

        step = asfloat(self.variables.step)
        beta1 = asfloat(self.beta1)
        beta2 = asfloat(self.beta2)
        epsilon = asfloat(self.epsilon)

        gradient = T.grad(self.variables.error_func, wrt=parameter)

        first_moment = (
            beta1 * prev_first_moment +
            asfloat(1. - beta1) * gradient)
        second_moment = (
            beta2 * prev_second_moment +
            asfloat(1. - beta2) * gradient ** 2
        )

        first_moment_bias_corrected = first_moment / (1. - beta1 ** epoch)
        second_moment_bias_corrected = second_moment / (1. - beta2 ** epoch)

        parameter_delta = first_moment_bias_corrected * (
            T.sqrt(second_moment_bias_corrected) + epsilon
        )

        return [
            (prev_first_moment, first_moment),
            (prev_second_moment, second_moment),
            (parameter, parameter - step * parameter_delta),
        ]
