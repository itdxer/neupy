import tensorflow as tf
import numpy as np

from neupy.utils import asfloat, tensorflow_session
from neupy.algorithms.gd import StepSelectionBuiltIn
from neupy.core.properties import BoundedProperty, ProperFractionProperty
from .base import GradientDescent


__all__ = ('RPROP', 'IRPROPPlus')


class RPROP(StepSelectionBuiltIn, GradientDescent):
    """
    Resilient backpropagation (RPROP) is an optimization
    algorithm for supervised learning.

    Parameters
    ----------
    minstep : float
        Minimum possible value for step. Defaults to ``0.1``.

    maxstep : float
        Maximum possible value for step. Defaults to ``50``.

    increase_factor : float
        Increase factor for step in case when gradient doesn't change
        sign compare to previous epoch.

    decrease_factor : float
        Decrease factor for step in case when gradient changes sign
        compare to previous epoch.

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
    >>> rpropnet = algorithms.RPROP((2, 3, 1))
    >>> rpropnet.train(x_train, y_train)

    See Also
    --------
    :network:`IRPROPPlus` : iRPROP+ algorithm.
    :network:`GradientDescent` : GradientDescent algorithm.
    """

    # This properties correct upper and lower bounds for steps.
    minstep = BoundedProperty(default=0.1, minval=0)
    maxstep = BoundedProperty(default=50, minval=0)

    # This properties increase/decrease step by deviding it to
    # some coeffitient.
    increase_factor = BoundedProperty(minval=1, default=1.2)
    decrease_factor = ProperFractionProperty(default=0.5)

    def init_prev_delta(self, parameter):
        self.prev_delta = tf.Variable(
            tf.zeros(parameter.shape),
            name="{}/prev-delta".format(parameter.op.name),
            dtype=tf.float32,
        )
        return self.prev_delta

    def init_param_updates(self, layer, parameter):
        prev_delta = self.init_prev_delta(parameter)
        steps = tf.Variable(
            tf.ones_like(parameter) * self.step,
            name="{}/steps".format(parameter.op.name),
            dtype=tf.float32,
        )
        prev_gradient = tf.Variable(
            tf.zeros(parameter.shape),
            name="{}/prev-grad".format(parameter.op.name),
            dtype=tf.float32,
        )

        gradient, = tf.gradients(self.variables.error_func, parameter)

        grad_product = prev_gradient * gradient
        negative_gradients = tf.less(grad_product, 0)

        updated_steps = tf.clip_by_value(
            tf.where(
                tf.greater(grad_product, 0),
                steps * self.increase_factor,
                tf.where(
                    negative_gradients,
                    steps * self.decrease_factor,
                    steps
                )
            ),
            self.minstep,
            self.maxstep,
        )
        parameter_delta = tf.where(
            negative_gradients,
            prev_delta,
            tf.where(
                tf.less(gradient, 0),
                -updated_steps,
                updated_steps,
            )
        )
        updated_prev_gradient = tf.where(
            negative_gradients,
            tf.zeros_like(gradient),
            gradient,
        )

        return [
            (parameter, parameter - parameter_delta),
            (steps, updated_steps),
            (prev_gradient, updated_prev_gradient),
            (self.prev_delta, -parameter_delta),
        ]


class IRPROPPlus(RPROP):
    """
    iRPROP+ is an optimization algorithm for supervised learning.
    This is a variation of the :network:`RPROP` algorithm.

    Parameters
    ----------
    {RPROP.minstep}

    {RPROP.maxstep}

    {RPROP.increase_factor}

    {RPROP.decrease_factor}

    {GradientDescent.addons}

    {ConstructibleNetwork.connection}

    {ConstructibleNetwork.error}

    {BaseNetwork.show_epoch}

    {BaseNetwork.shuffle_data}

    {BaseNetwork.epoch_end_signal}

    {BaseNetwork.train_end_signal}

    {Verbose.verbose}

    Methods
    -------
    {BaseSkeleton.predict}

    {ConstructibleNetwork.train}

    {BaseSkeleton.fit}

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms
    >>>
    >>> x_train = np.array([[1, 2], [3, 4]])
    >>> y_train = np.array([[1], [0]])
    >>>
    >>> rpropnet = algorithms.IRPROPPlus((2, 3, 1))
    >>> rpropnet.train(x_train, y_train)

    See Also
    --------
    :network:`RPROP` : RPROP algorithm.
    :network:`GradientDescent` : GradientDescent algorithm.
    """
    def init_variables(self):
        super(IRPROPPlus, self).init_variables()
        self.variables.update(
            last_error=tf.Variable(np.nan, name='irprop-plus/last-error'),
            previous_error=tf.Variable(np.nan, name='irprop-plus/previous-error'),
        )

    def on_epoch_start_update(self, epoch):
        super(IRPROPPlus, self).on_epoch_start_update(epoch)

        previous_error = self.errors.previous()
        if previous_error:
            last_error = self.errors.last()
            session = tensorflow_session()

            self.variables.last_error.load(last_error, session)
            self.variables.previous_error.load(previous_error, session)

    def init_prev_delta(self, parameter):
        prev_delta = super(IRPROPPlus, self).init_prev_delta(parameter)

        last_error = self.variables.last_error
        prev_error = self.variables.previous_error

        return tf.where(
            last_error > prev_error,
            prev_delta,
            tf.zeros_like(prev_delta),
        )
