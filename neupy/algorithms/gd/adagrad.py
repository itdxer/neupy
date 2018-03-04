import tensorflow as tf
import numpy as np

from neupy.utils import asfloat
from neupy.core.properties import NumberProperty
from .base import MinibatchGradientDescent


__all__ = ('Adagrad',)


class Adagrad(MinibatchGradientDescent):
    """
    Adagrad algorithm.

    Parameters
    ----------
    epsilon : float
        Value need to be greater than ``0``.
        Defaults to ``1e-5``.

    {MinibatchGradientDescent.Parameters}

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
    >>> mnet = algorithms.Adagrad((2, 3, 1))
    >>> mnet.train(x_train, y_train)
    """
    epsilon = NumberProperty(default=1e-5, minval=0)

    def init_param_updates(self, layer, parameter):
        step = self.variables.step
        prev_mean_squred_grad = tf.Variable(
            tf.zeros(parameter.shape),
            name="{}/prev-mean-squred-grad".format(parameter.op.name),
            dtype=tf.float32,
        )

        gradient, = tf.gradients(self.variables.error_func, parameter)

        mean_squred_grad = prev_mean_squred_grad + gradient ** 2
        parameter_delta = gradient * tf.sqrt(mean_squred_grad + self.epsilon)

        return [
            (prev_mean_squred_grad, mean_squred_grad),
            (parameter, parameter - step * parameter_delta),
        ]
