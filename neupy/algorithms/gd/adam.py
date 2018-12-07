import tensorflow as tf

from neupy.utils import asfloat
from neupy.core.properties import ProperFractionProperty, NumberProperty
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

    {BaseGradientDescent.addons}

    {ConstructibleNetwork.connection}

    {ConstructibleNetwork.error}

    {BaseNetwork.show_epoch}

    {BaseNetwork.shuffle_data}

    {BaseNetwork.epoch_end_signal}

    {BaseNetwork.train_end_signal}

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
    >>>
    >>> x_train = np.array([[1, 2], [3, 4]])
    >>> y_train = np.array([[1], [0]])
    >>>
    >>> mnet = algorithms.Adam((2, 3, 1))
    >>> mnet.train(x_train, y_train)
    """
    step = NumberProperty(default=0.001, minval=0)
    beta1 = ProperFractionProperty(default=0.9)
    beta2 = ProperFractionProperty(default=0.999)
    epsilon = NumberProperty(default=1e-7, minval=0)

    def init_variables(self):
        super(Adam, self).init_variables()

        self.variables.iteration = tf.Variable(
            asfloat(1),
            name='iteration',
            dtype=tf.float32,
        )

    def init_train_updates(self):
        updates = []

        iteration = self.variables.iteration
        step = self.variables.step

        # Since beta1 and beta2 are typically close to 1 and initial
        # values for first and second moments are close to zero the
        # initial estimates for these moments will be biased towards zero.
        # In order to solve this problem we need to correct this bias
        # by rescaling moments with large values during first updates
        # and vanishing this scaling factor more and more after every
        # update.
        #
        # Note that bias correction factor has been changed in order
        # to improve computational speed (suggestion from the original
        # paper).
        bias_correction = (
            tf.sqrt(1. - self.beta2 ** iteration) /
            (1. - self.beta1 ** iteration)
        )

        for layer, parameter, gradient in self.iter_params_and_grads():
            prev_first_moment = tf.Variable(
                tf.zeros(parameter.shape),
                name="{}/prev-first-moment".format(parameter.op.name),
                dtype=tf.float32,
            )
            prev_second_moment = tf.Variable(
                tf.zeros(parameter.shape),
                name="{}/prev-second-moment".format(parameter.op.name),
                dtype=tf.float32,
            )

            first_moment = (
                self.beta1 * prev_first_moment +
                (1. - self.beta1) * gradient
            )
            second_moment = (
                self.beta2 * prev_second_moment +
                (1. - self.beta2) * gradient ** 2
            )

            parameter_delta = bias_correction * first_moment / (
                tf.sqrt(second_moment) + self.epsilon)

            updates.extend([
                (prev_first_moment, first_moment),
                (prev_second_moment, second_moment),
                (parameter, parameter - step * parameter_delta),
            ])

        updates.append((iteration, iteration + 1))
        return updates
