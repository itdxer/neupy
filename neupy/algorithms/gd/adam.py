import tensorflow as tf

from neupy.core.properties import (
    ProperFractionProperty,
    ScalarVariableProperty,
    NumberProperty,
)
from .base import GradientDescent


__all__ = ('Adam',)


class Adam(GradientDescent):
    """
    Adam algorithm.

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

    {GradientDescent.batch_size}

    {BaseOptimizer.regularizer}

    {BaseOptimizer.network}

    {BaseOptimizer.loss}

    {BaseNetwork.show_epoch}

    {BaseNetwork.shuffle_data}

    {BaseNetwork.signals}

    {Verbose.verbose}

    Attributes
    ----------
    {GradientDescent.Attributes}

    Methods
    -------
    {GradientDescent.Methods}

    References
    ----------
    [1] Diederik P. Kingma, Jimmy Lei Ba
        Adam: a Method for Stochastic Optimization.
        https://arxiv.org/pdf/1412.6980.pdf

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
    >>> optimizer = algorithms.Adam(network)
    >>> optimizer.train(x_train, y_train)
    """
    step = ScalarVariableProperty(default=0.001)
    beta1 = ProperFractionProperty(default=0.9)
    beta2 = ProperFractionProperty(default=0.999)
    epsilon = NumberProperty(default=1e-7, minval=0)

    def init_train_updates(self):
        optimizer = tf.train.AdamOptimizer(
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            learning_rate=self.step,
        )
        self.functions.optimizer = optimizer
        return [optimizer.minimize(self.variables.loss)]
