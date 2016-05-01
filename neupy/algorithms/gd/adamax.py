import theano
import theano.tensor as T
import numpy as np

from neupy.utils import asfloat
from neupy.core.properties import ProperFractionProperty, NumberProperty
from .base import MinibatchGradientDescent


__all__ = ('Adamax',)


class Adamax(MinibatchGradientDescent):
    """ AdaMax algorithm.

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
    epsilon = NumberProperty(default=1e-8, minval=0)

    def init_layers(self):
        super(Adamax, self).init_layers()
        for layer in self.layers:
            for parameter in layer.parameters:
                parameter_shape = T.shape(parameter).eval()
                parameter.prev_first_moment = theano.shared(
                    name="prev_first_moment_" + parameter.name,
                    value=asfloat(np.zeros(parameter_shape)),
                )
                parameter.prev_weighted_inf_norm = theano.shared(
                    name="prev_weighted_inf_norm_" + parameter.name,
                    value=asfloat(np.zeros(parameter_shape)),
                )

    def init_param_updates(self, layer, parameter):
        epoch = self.variables.epoch
        prev_first_moment = parameter.prev_first_moment
        prev_weighted_inf_norm = parameter.prev_weighted_inf_norm

        step = self.variables.step
        beta1 = self.beta1
        beta2 = self.beta2

        gradient = T.grad(self.variables.error_func, wrt=parameter)

        first_moment = beta1 * prev_first_moment + (1 - beta1) * gradient
        weighted_inf_norm = T.maximum(beta2 * prev_weighted_inf_norm,
                                      T.abs_(gradient))

        parameter_delta = (
            (1 / (1 - beta1 ** epoch)) *
            (first_moment / (weighted_inf_norm + self.epsilon))
        )

        return [
            (prev_first_moment, first_moment),
            (prev_weighted_inf_norm, weighted_inf_norm),
            (parameter, parameter - step * parameter_delta),
        ]
