import tensorflow as tf
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
    epsilon = NumberProperty(default=1e-7, minval=0)

    def init_param_updates(self, layer, parameter):
        epoch = self.variables.epoch
        step = self.variables.step
        beta1 = self.beta1
        beta2 = self.beta2

        prev_first_moment = tf.get_variable(
            "{}/prev-first-moment".format(parameter.op.name),
            parameter.shape,
            dtype=tf.float32,
            initializer=tf.zeros_initializer,
        )
        prev_weighted_inf_norm = tf.get_variable(
            "{}/prev-weighted-inf-norm".format(parameter.op.name),
            parameter.shape,
            dtype=tf.float32,
            initializer=tf.zeros_initializer,
        )
        gradient, = tf.gradients(self.variables.error_func, parameter)

        first_moment = beta1 * prev_first_moment + (1. - beta1) * gradient
        weighted_inf_norm = tf.maximum(
            beta2 * prev_weighted_inf_norm,
            tf.abs(gradient),
        )

        parameter_delta = (
            (step / (1. - tf.pow(beta1, epoch + 1))) *
            (first_moment / (weighted_inf_norm + self.epsilon))
        )

        return [
            (prev_first_moment, first_moment),
            (prev_weighted_inf_norm, weighted_inf_norm),
            (parameter, parameter - parameter_delta),
        ]
