import tensorflow as tf
import numpy as np

from neupy.utils import tensorflow_session
from neupy.algorithms.gd import StepSelectionBuiltIn
from neupy.core.properties import BoundedProperty, ProperFractionProperty
from .base import BaseGradientDescent


__all__ = ('RPROP', 'IRPROPPlus')


class RPROP(StepSelectionBuiltIn, BaseGradientDescent):
    """
    Resilient backpropagation (RPROP) is an optimization
    algorithm for supervised learning.

    RPROP algorithm takes into account only direction of the gradient
    and completely ignores its magnitude. Every weight values has a unique
    step size associated with it (by default all of the are equal to ``step``).

    The rule is following, when gradient direction changes (sign of the
    gradient) we decrease step size for specific weight multiplying it by
    ``decrease_factor`` and if sign stays the same than we increase step
    size for this specific weight multiplying it by ``increase_factor``.

    The step size is always bounded by ``minstep`` and ``maxstep``.

    Notes
    -----
    Algorithm doesn't work with mini-batches.

    Parameters
    ----------
    minstep : float
        Minimum possible value for step. Defaults to ``0.001``.

    maxstep : float
        Maximum possible value for step. Defaults to ``10``.

    increase_factor : float
        Increase factor for step in case when gradient doesn't change
        sign compare to previous epoch.

    decrease_factor : float
        Decrease factor for step in case when gradient changes sign
        compare to previous epoch.

    {BaseGradientDescent.Parameters}

    Attributes
    ----------
    {BaseGradientDescent.Attributes}

    Methods
    -------
    {BaseGradientDescent.Methods}

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
    minstep = BoundedProperty(default=0.001, minval=0)
    maxstep = BoundedProperty(default=10, minval=0)

    # This properties increase/decrease step by deviding it to
    # some coeffitient.
    increase_factor = BoundedProperty(minval=1, default=1.2)
    decrease_factor = ProperFractionProperty(default=0.5)

    def update_prev_delta(self, prev_delta):
        return prev_delta

    def init_train_updates(self):
        updates = []

        for layer, parameter, gradient in self.iter_params_and_grads():
            with tf.variable_scope(parameter.op.name):
                steps = tf.Variable(
                    # Steps will be decreased after the first iteration,
                    # because all previous gradients are equal to zero.
                    # In order to make sure that network will use the same
                    # step per every weight we re-scale step and after the
                    # first iteration it will be multiplied by
                    # ``decrease_factor`` and scaled back to the default
                    # step value.
                    tf.ones_like(parameter) * self.step,
                    name="steps",
                    dtype=tf.float32,
                )
                prev_delta = tf.Variable(
                    tf.zeros(parameter.shape),
                    name="prev-delta",
                    dtype=tf.float32,
                )
                # We collect only signs since it ensures numerical stability
                # after multiplication when we deal with small numbers.
                prev_gradient_sign = tf.Variable(
                    tf.zeros(parameter.shape),
                    name="prev-grad-sign",
                    dtype=tf.float32,
                )

            updated_prev_delta = self.update_prev_delta(prev_delta)
            gradient_sign = tf.sign(gradient)

            grad_sign_product = gradient_sign * prev_gradient_sign
            gradient_changed_sign = tf.equal(grad_sign_product, -1)

            updated_steps = tf.clip_by_value(
                tf.where(
                    tf.equal(grad_sign_product, 1),
                    steps * self.increase_factor,
                    tf.where(
                        gradient_changed_sign,
                        steps * self.decrease_factor,
                        steps,
                    )
                ),
                self.minstep,
                self.maxstep,
            )
            parameter_delta = tf.where(
                gradient_changed_sign,
                # If we subtract previous negative weight update it means
                # that we will revert weight update that has been  applied
                # in the previous iteration.
                -updated_prev_delta,
                updated_steps * gradient_sign,
            )
            # Making sure that during the next iteration sign, after
            # we multiplied by the new gradient, won't be negative.
            # Otherwise, the same roll back using previous delta
            # won't make much sense.
            clipped_gradient_sign = tf.where(
                gradient_changed_sign,
                tf.zeros_like(gradient_sign),
                gradient_sign,
            )

            updates.extend([
                (parameter, parameter - parameter_delta),
                (steps, updated_steps),
                (prev_gradient_sign, clipped_gradient_sign),
                (prev_delta, parameter_delta),
            ])

        return updates


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

    {BaseGradientDescent.addons}

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

    Notes
    -----
    {RPROP.Notes}

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

    References
    ----------
    [1] Christian Igel, Michael Huesken (2000)
        Improving the Rprop Learning Algorithm

    See Also
    --------
    :network:`RPROP` : RPROP algorithm.
    :network:`GradientDescent` : GradientDescent algorithm.
    """
    def init_variables(self):
        super(IRPROPPlus, self).init_variables()
        self.variables.update(
            last_error=tf.Variable(np.nan, name='irprop-plus/last-error'),
            previous_error=tf.Variable(
                np.nan, name='irprop-plus/previous-error'),
        )

    def on_epoch_start_update(self, epoch):
        super(IRPROPPlus, self).on_epoch_start_update(epoch)

        previous_error = self.errors.previous()
        if previous_error:
            last_error = self.errors.last()
            session = tensorflow_session()

            self.variables.last_error.load(last_error, session)
            self.variables.previous_error.load(previous_error, session)

    def update_prev_delta(self, prev_delta):
        last_error = self.variables.last_error
        prev_error = self.variables.previous_error

        return tf.where(
            # We revert weight when gradient changed the sign only in
            # cases when error increased. Otherwise we don't apply any
            # update for this weight.
            last_error > prev_error,
            prev_delta,
            tf.zeros_like(prev_delta),
        )
