import tensorflow as tf

from neupy.core.properties import ProperFractionProperty, Property
from .base import GradientDescent


__all__ = ('Momentum',)


class Momentum(GradientDescent):
    """
    Momentum algorithm.

    Parameters
    ----------
    momentum : float
        Control previous gradient ratio. Defaults to ``0.9``.

    nesterov : bool
        Instead of classic momentum computes Nesterov momentum.
        Defaults to ``False``.

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
    >>> mnet = algorithms.Momentum((2, 3, 1))
    >>> mnet.train(x_train, y_train)

    See Also
    --------
    :network:`GradientDescent` : GradientDescent algorithm.
    """
    momentum = ProperFractionProperty(default=0.9)
    nesterov = Property(default=False, expected_type=bool)

    def init_train_updates(self):
        """
        Initialize updates that would be applied after
        each training epoch.
        """
        updates = []
        step = self.variables.step

        for layer, parameter, gradient in self.iter_params_and_grads():
            previous_velocity = tf.Variable(
                tf.zeros(parameter.shape),
                name="{}/previous-velocity".format(parameter.op.name),
                dtype=tf.float32,
            )
            velocity = self.momentum * previous_velocity - step * gradient

            if self.nesterov:
                velocity = self.momentum * velocity - step * gradient

            updates.extend([
                (parameter, parameter + velocity),
                (previous_velocity, velocity),
            ])

        return updates
