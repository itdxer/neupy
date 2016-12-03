import theano
import theano.tensor as T
import numpy as np

from neupy.utils import asfloat
from neupy.core.properties import ProperFractionProperty, NumberProperty
from .base import MinibatchGradientDescent


__all__ = ('Adamax',)


class Adamax(MinibatchGradientDescent):
    """
    AdaMax algorithm.

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

    {ConstructibleNetwork.connection}

    {ConstructibleNetwork.error}

    {BaseNetwork.show_epoch}

    {BaseNetwork.shuffle_data}

    {BaseNetwork.epoch_end_signal}

    {BaseNetwork.train_end_signal}

    {Verbose.verbose}

    Attributes
    ----------
    {MinibatchGradientDescent.Attributes}

    Methods
    -------
    {MinibatchGradientDescent.Methods}

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms
    >>>
    >>> x_train = np.array([[1, 2], [3, 4]])
    >>> y_train = np.array([[1], [0]])
    >>>
    >>> mnet = algorithms.Adamax((2, 3, 1))
    >>> mnet.train(x_train, y_train)
    """
    step = NumberProperty(default=0.001, minval=0)
    beta1 = ProperFractionProperty(default=0.9)
    beta2 = ProperFractionProperty(default=0.999)
    epsilon = NumberProperty(default=1e-8, minval=0)

    def init_param_updates(self, layer, parameter):
        epoch = self.variables.epoch
        step = self.variables.step
        beta1 = self.beta1
        beta2 = self.beta2

        parameter_shape = T.shape(parameter).eval()
        prev_first_moment = theano.shared(
            name="{}/prev-first-moment".format(parameter.name),
            value=asfloat(np.zeros(parameter_shape)),
        )
        prev_weighted_inf_norm = theano.shared(
            name="{}/prev-weighted-inf-norm".format(parameter.name),
            value=asfloat(np.zeros(parameter_shape)),
        )

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
