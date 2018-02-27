import tensorflow as tf
import numpy as np

from neupy.utils import asfloat
from neupy.core.properties import ProperFractionProperty, Property
from .base import MinibatchGradientDescent


__all__ = ('Momentum',)


class Momentum(MinibatchGradientDescent):
    """
    Momentum algorithm.

    Parameters
    ----------
    momentum : float
        Control previous gradient ratio. Defaults to ``0.9``.

    nesterov : bool
        Instead of classic momentum computes Nesterov momentum.
        Defaults to ``False``.

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
    >>> mnet = algorithms.Momentum((2, 3, 1))
    >>> mnet.train(x_train, y_train)

    See Also
    --------
    :network:`GradientDescent` : GradientDescent algorithm.
    """
    momentum = ProperFractionProperty(default=0.9)
    nesterov = Property(default=False, expected_type=bool)

    def init_param_updates(self, layer, parameter):
        step = self.variables.step
        previous_velocity = tf.get_variable(
            "{}/previous-velocity".format(parameter.op.name),
            parameter.shape,
            dtype=tf.float32,
            initializer=tf.zeros_initializer,
        )

        gradient, = tf.gradients(self.variables.error_func, parameter)
        velocity = self.momentum * previous_velocity - step * gradient

        if self.nesterov:
            velocity = self.momentum * velocity - step * gradient

        return [
            (parameter, parameter + velocity),
            (previous_velocity, velocity),
        ]
