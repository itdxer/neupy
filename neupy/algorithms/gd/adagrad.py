import tensorflow as tf

from .base import GradientDescent


__all__ = ('Adagrad',)


class Adagrad(GradientDescent):
    """
    Adagrad algorithm.

    Parameters
    ----------
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
    >>> optimizer = algorithms.Adagrad(network)
    >>> optimizer.train(x_train, y_train)

    References
    ----------
    [1] John Duchi, Elad Hazan, Yoram Singer,
        Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization
        http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
    """
    def init_train_updates(self):
        optimizer = tf.train.AdagradOptimizer(
            learning_rate=self.step,
        )
        self.functions.optimizer = optimizer
        return [optimizer.minimize(self.variables.loss)]
