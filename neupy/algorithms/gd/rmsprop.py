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

    momentum : float
        Defaults to ``0``.

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
    >>> from neupy.layers import *
    >>>
    >>> x_train = np.array([[1, 2], [3, 4]])
    >>> y_train = np.array([[1], [0]])
    >>>
    >>> network = Input(2) >> Sigmoid(3) >> Sigmoid(1)
    >>> optimizer = algorithms.RMSProp(network)
    >>> optimizer.train(x_train, y_train)
    """
    decay = ProperFractionProperty(default=0.95)
    momentum = NumberProperty(default=0, minval=0)
    epsilon = NumberProperty(default=1e-7, minval=0)

    def init_train_updates(self):
        optimizer = tf.train.RMSPropOptimizer(
            decay=self.decay,
            momentum=self.momentum,
            epsilon=self.epsilon,
            learning_rate=self.step,
        )
        self.functions.optimizer = optimizer
        return [optimizer.minimize(self.variables.loss)]
