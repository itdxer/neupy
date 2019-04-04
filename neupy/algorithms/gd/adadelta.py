import tensorflow as tf

from neupy.core.properties import (
    ProperFractionProperty,
    ScalarVariableProperty,
    NumberProperty,
)
from .base import GradientDescent


__all__ = ('Adadelta',)


class Adadelta(GradientDescent):
    """
    Adadelta algorithm.

    Parameters
    ----------
    rho : float
        Decay rate. Value need to be between ``0``
        and ``1``. Defaults to ``0.95``.

    epsilon : float
        Value need to be greater than ``0``. Defaults to ``1e-7``.

    step : float
        Learning rate, defaults to ``1.0``. Original paper doesn't have
        learning rate specified in the paper. Step value equal to ``1.0``
        allow to achieve the same effect, since multiplication by one won't
        have any effect on the update.

    {GradientDescent.batch_size}

    {BaseOptimizer.regularizer}

    {BaseOptimizer.network}

    {BaseOptimizer.loss}

    {BaseNetwork.show_epoch}

    {BaseNetwork.shuffle_data}

    {BaseNetwork.signals}

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
    >>> optimizer = algorithms.Adadelta(network)
    >>> optimizer.train(x_train, y_train)

    References
    ----------
    [1] Matthew D. Zeiler,
        ADADELTA: An Adaptive Learning Rate Method
        https://arxiv.org/pdf/1212.5701.pdf
    """
    step = ScalarVariableProperty(default=1.0)
    rho = ProperFractionProperty(default=0.95)
    epsilon = NumberProperty(default=1e-7, minval=0)

    def init_train_updates(self):
        optimizer = tf.train.AdadeltaOptimizer(
            rho=self.rho,
            epsilon=self.epsilon,
            learning_rate=self.step,
        )
        self.functions.optimizer = optimizer
        return [optimizer.minimize(self.variables.loss)]
