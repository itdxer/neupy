import tensorflow as tf

from neupy.core.properties import ProperFractionProperty, NumberProperty
from .base import GradientDescent


__all__ = ('Adadelta',)


class Adadelta(GradientDescent):
    """
    Adadelta algorithm.

    Parameters
    ----------
    decay : float
        Decay rate. Value need to be between ``0``
        and ``1``. Defaults to ``0.95``.

    epsilon : float
        Value need to be greater than ``0``. Defaults to ``1e-7``.

    step : float
        Learning rate, defaults to ``1.0``. Original paper doesn't have
        learning rate specified in the paper. Step value equal to ``1.0``
        allow to achive the same effect, since multiplication by one won't
        have any effect on the update.

    {GradientDescent.batch_size}

    {BaseGradientDescent.addons}

    {ConstructibleNetwork.connection}

    {ConstructibleNetwork.error}

    {BaseNetwork.show_epoch}

    {BaseNetwork.shuffle_data}

    {BaseNetwork.epoch_end_signal}

    {BaseNetwork.train_end_signal}

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
    >>> mnet = algorithms.Adadelta((2, 3, 1))
    >>> mnet.train(x_train, y_train)

    References
    ----------
    [1] Matthew D. Zeiler,
        ADADELTA: An Adaptive Learning Rate Method
        https://arxiv.org/pdf/1212.5701.pdf
    """
    step = NumberProperty(default=1.0, minval=0)
    decay = ProperFractionProperty(default=0.95)
    epsilon = NumberProperty(default=1e-7, minval=0)

    def init_train_updates(self):
        updates = []
        step = self.variables.step
        epsilon = self.epsilon

        for layer, parameter, gradient in self.iter_params_and_grads():
            prev_mean_squred_grad = tf.Variable(
                tf.zeros(parameter.shape),
                name="{}/prev-mean-squred-grad".format(parameter.op.name),
                dtype=tf.float32,
            )
            prev_mean_squared_update = tf.Variable(
                tf.zeros(parameter.shape),
                name="{}/prev-mean-squred-update".format(parameter.op.name),
                dtype=tf.float32,
            )

            mean_squred_grad = (
                self.decay * prev_mean_squred_grad +
                (1 - self.decay) * gradient ** 2
            )
            parameter_delta = gradient * (
                tf.sqrt(prev_mean_squared_update + epsilon) /
                tf.sqrt(mean_squred_grad + epsilon)
            )
            mean_squared_update = (
                self.decay * prev_mean_squared_update +
                (1 - self.decay) * parameter_delta ** 2
            )

            updates.extend([
                (prev_mean_squred_grad, mean_squred_grad),
                (prev_mean_squared_update, mean_squared_update),
                (parameter, parameter - step * parameter_delta),
            ])

        return updates
