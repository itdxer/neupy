import tensorflow as tf

from neupy.core.properties import NumberProperty
from .base import GradientDescent


__all__ = ('Adagrad',)


class Adagrad(GradientDescent):
    """
    Adagrad algorithm.

    Parameters
    ----------
    epsilon : float
        Value need to be greater than ``0``.
        Defaults to ``1e-5``.

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
    >>> mnet = algorithms.Adagrad((2, 3, 1))
    >>> mnet.train(x_train, y_train)

    References
    ----------
    [1] John Duchi, Elad Hazan, Yoram Singer,
        Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization
        http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
    """
    epsilon = NumberProperty(default=1e-5, minval=0)

    def init_train_updates(self):
        updates = []
        step = self.variables.step

        for layer, parameter, gradient in self.iter_params_and_grads():
            prev_mean_squred_grad = tf.Variable(
                tf.zeros(parameter.shape),
                name="{}/prev-mean-squred-grad".format(parameter.op.name),
                dtype=tf.float32,
            )

            mean_squred_grad = prev_mean_squred_grad + gradient ** 2
            parameter_delta = gradient / (
                tf.sqrt(mean_squred_grad + self.epsilon))

            updates.extend([
                (prev_mean_squred_grad, mean_squred_grad),
                (parameter, parameter - step * parameter_delta),
            ])

        return updates
