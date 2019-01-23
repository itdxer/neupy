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
    >>> from neupy.layers import *
    >>>
    >>> x_train = np.array([[1, 2], [3, 4]])
    >>> y_train = np.array([[1], [0]])
    >>>
    >>> network = Input(2) >> Sigmoid(3) >> Sigmoid(1)
    >>> optimizer = algorithms.Momentum(network)
    >>> optimizer.train(x_train, y_train)

    See Also
    --------
    :network:`GradientDescent` : GradientDescent algorithm.
    """
    momentum = ProperFractionProperty(default=0.9)
    nesterov = Property(default=False, expected_type=bool)

    def init_train_updates(self):
        optimizer = tf.train.MomentumOptimizer(
            use_nesterov=self.nesterov,
            momentum=self.momentum,
            learning_rate=self.step,
        )
        self.functions.optimizer = optimizer
        return [optimizer.minimize(self.variables.loss)]
