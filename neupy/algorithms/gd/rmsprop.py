import tensorflow as tf

from neupy.core.properties import ProperFractionProperty, NumberProperty
from .base import GradientDescent


__all__ = ('RMSProp',)


class RMSProp(GradientDescent):
    """
    RMSProp algorithm.

    Parameters
    ----------
    decay : float
        Decay rate. Value need to be between ``0`` and ``1``.
        Defaults to ``0.95``.

    epsilon : float
        Value need to be greater than ``0``. Defaults to ``1e-7``.

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
    >>> mnet = algorithms.RMSProp((2, 3, 1))
    >>> mnet.train(x_train, y_train)
    """
    decay = ProperFractionProperty(default=0.95)
    epsilon = NumberProperty(default=1e-7, minval=0)

    def init_train_updates(self):
        updates = []
        step = self.variables.step

        for layer, parameter, gradient in self.iter_params_and_grads():
            prev_mean_squred_grad = tf.Variable(
                tf.zeros(parameter.shape),
                name="{}/prev-mean-squared-grad".format(parameter.op.name),
                dtype=tf.float32,
            )

            mean_squred_grad = (
                self.decay * prev_mean_squred_grad +
                (1. - self.decay) * tf.pow(gradient, 2)
            )
            parameter_delta = gradient / tf.sqrt(
                mean_squred_grad + self.epsilon)

            updates.extend([
                (prev_mean_squred_grad, mean_squred_grad),
                (parameter, parameter - step * parameter_delta),
            ])

        return updates
