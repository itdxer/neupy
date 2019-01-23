import tensorflow as tf

from neupy.utils import asfloat
from neupy.core.properties import (
    ProperFractionProperty,
    ScalarVariableProperty,
    NumberProperty,
)
from .base import GradientDescent


__all__ = ('Adamax',)


class Adamax(GradientDescent):
    """
    AdaMax algorithm.

    Parameters
    ----------
    beta1 : float
        Decay rate. Value need to be between ``0`` and ``1``.
        Defaults to ``0.9``.

    beta2 : float
        Decay rate. Value need to be between ``0`` and ``1``.
        Defaults to ``0.999``.

    epsilon : float
        Value need to be greater than ``0``. Defaults to ``1e-7``.

    step : float
        Learning rate, defaults to ``0.002``.

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
    >>> mnet = algorithms.Adamax(network)
    >>> mnet.train(x_train, y_train)

    References
    ----------
    [1] Diederik P. Kingma, Jimmy Lei Ba
        Adam: a Method for Stochastic Optimization.
        https://arxiv.org/pdf/1412.6980.pdf
    """
    step = ScalarVariableProperty(default=0.002)
    beta1 = ProperFractionProperty(default=0.9)
    beta2 = ProperFractionProperty(default=0.999)
    epsilon = NumberProperty(default=1e-7, minval=0)

    def init_functions(self):
        self.variables.iteration = tf.Variable(
            asfloat(1),
            name='iteration',
            dtype=tf.float32,
        )
        super(Adamax, self).init_functions()

    def init_train_updates(self):
        iteration = self.variables.iteration
        beta1 = self.beta1
        beta2 = self.beta2

        updates = []
        variables = []

        for (_, _), variable in self.network.variables.items():
            if variable.trainable:
                variables.append(variable)

        gradients = tf.gradients(self.variables.loss, variables)
        scale = self.step / (1. - beta1 ** iteration)

        for parameter, gradient in zip(variables, gradients):
            prev_first_moment = tf.Variable(
                tf.zeros(parameter.shape),
                name="{}/prev-first-moment".format(parameter.op.name),
                dtype=tf.float32,
            )
            prev_weighted_inf_norm = tf.Variable(
                tf.zeros(parameter.shape),
                name="{}/prev-weighted-inf-norm".format(parameter.op.name),
                dtype=tf.float32,
            )

            first_moment = beta1 * prev_first_moment + (1. - beta1) * gradient
            weighted_inf_norm = tf.maximum(
                beta2 * prev_weighted_inf_norm,
                tf.abs(gradient),
            )

            parameter_delta = (
                scale * (first_moment / (weighted_inf_norm + self.epsilon)))

            updates.extend([
                (prev_first_moment, first_moment),
                (prev_weighted_inf_norm, weighted_inf_norm),
                (parameter, parameter - parameter_delta),
            ])

        updates.append((iteration, iteration + 1))
        return updates
