from __future__ import division

import tensorflow as tf

from neupy.core.properties import BoundedProperty
from .base import GradientDescent


__all__ = ('Quickprop',)


class Quickprop(GradientDescent):
    """
    Quickprop :network:`GradientDescent` algorithm optimization.

    Parameters
    ----------
    upper_bound : float
        Maximum possible value for weight update.
        Defaults to ``1``.

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
    >>> qpnet = algorithms.Quickprop((2, 3, 1))
    >>> qpnet.train(x_train, y_train)

    See Also
    --------
    :network:`GradientDescent` : GradientDescent algorithm.
    """
    upper_bound = BoundedProperty(default=1, minval=0)

    def init_train_updates(self):
        updates = []
        step = self.variables.step

        for layer, parameter, gradient in self.iter_params_and_grads():
            prev_delta = tf.Variable(
                tf.zeros(parameter.shape),
                name="{}/prev-delta".format(parameter.op.name),
                dtype=tf.float32,
            )
            prev_gradient = tf.Variable(
                tf.zeros(parameter.shape),
                name="{}/prev-grad".format(parameter.op.name),
                dtype=tf.float32,
            )
            grad_delta = tf.abs(prev_gradient - gradient)

            parameter_delta = tf.where(
                tf.equal(self.variables.epoch, 1),
                gradient,
                tf.clip_by_value(
                    tf.abs(prev_delta) * gradient / grad_delta,
                    -self.upper_bound,
                    self.upper_bound
                )
            )
            updates.extend([
                (parameter, parameter - step * parameter_delta),
                (prev_gradient, gradient),
                (prev_delta, parameter_delta),
            ])

        return updates
