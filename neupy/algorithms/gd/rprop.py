import theano
import theano.tensor as T
import numpy as np

from neupy.utils import asfloat
from neupy.algorithms.gd import StepSelectionBuiltIn
from neupy.core.properties import BoundedProperty, ProperFractionProperty
from .base import GradientDescent


__all__ = ('RPROP', 'IRPROPPlus')


class RPROP(StepSelectionBuiltIn, GradientDescent):
    """ RPROP :network:`GradientDescent` algorithm optimization.

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
    {GradientDescent.addons}
    {ConstructableNetwork.connection}
    {ConstructableNetwork.error}
    {BaseNetwork.show_epoch}
    {BaseNetwork.shuffle_data}
    {BaseNetwork.epoch_end_signal}
    {BaseNetwork.train_end_signal}
    {Verbose.verbose}

    Methods
    -------
    {BaseSkeleton.predict}
    {SupervisedLearning.train}
    {BaseSkeleton.fit}

    Examples
    --------
    Simple example

    >>> import numpy as np
    >>> from neupy import algorithms
    >>>
    >>> x_train = np.array([[1, 2], [3, 4]])
    >>> y_train = np.array([[1], [0]])
    >>>
    >>> rpropnet = algorithms.RPROP(
    ...     (2, 3, 1),
    ...     verbose=False
    ... )
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

    def init_layers(self):
        super(RPROP, self).init_layers()
        for layer in self.layers:
            for parameter in layer.parameters:
                parameter_shape = T.shape(parameter).eval()
                parameter.prev_delta = theano.shared(
                    name="prev_delta_" + parameter.name,
                    value=asfloat(np.zeros(parameter_shape)),
                )
                parameter.prev_gradient = theano.shared(
                    name="prev_grad_" + parameter.name,
                    value=asfloat(np.zeros(parameter_shape)),
                )
                parameter.steps = theano.shared(
                    name="steps_{}" + parameter.name,
                    value=asfloat(np.ones(parameter_shape) * self.step),
                )

    def init_prev_delta(self, parameter):
        return parameter.prev_delta

    def init_param_updates(self, layer, parameter):
        gradient = T.grad(self.variables.error_func, wrt=parameter)

        steps = parameter.steps
        prev_delta = self.init_prev_delta(parameter)
        prev_gradient = parameter.prev_gradient

        grad_product = prev_gradient * gradient
        negative_gradients = T.lt(grad_product, 0)

        updated_steps = T.clip(
            T.switch(
                T.gt(grad_product, 0),
                steps * self.increase_factor,
                T.switch(
                    negative_gradients,
                    steps * self.decrease_factor,
                    steps
                )
            ),
            self.minstep,
            self.maxstep,
        )
        gradient_signs = T.switch(T.lt(gradient, 0), -1, 1)
        parameter_delta = T.switch(
            negative_gradients,
            prev_delta,
            gradient_signs * updated_steps
        )
        updated_prev_gradient = T.switch(negative_gradients, 0, gradient)

        return [
            (parameter, parameter - parameter_delta),
            (steps, updated_steps),
            (prev_gradient, updated_prev_gradient),
            (parameter.prev_delta, -parameter_delta),
        ]


class IRPROPPlus(RPROP):
    """ iRPROP+ :network:`GradientDescent` algorithm optimization.

    Parameters
    ----------
    {RPROP.minstep}
    {RPROP.maxstep}
    {RPROP.increase_factor}
    {RPROP.decrease_factor}
    {GradientDescent.addons}
    {ConstructableNetwork.connection}
    {ConstructableNetwork.error}
    {BaseNetwork.show_epoch}
    {BaseNetwork.shuffle_data}
    {BaseNetwork.epoch_end_signal}
    {BaseNetwork.train_end_signal}
    {Verbose.verbose}

    Methods
    -------
    {BaseSkeleton.predict}
    {SupervisedLearning.train}
    {BaseSkeleton.fit}

    Examples
    --------
    Simple example

    >>> import numpy as np
    >>> from neupy import algorithms
    >>>
    >>> x_train = np.array([[1, 2], [3, 4]])
    >>> y_train = np.array([[1], [0]])
    >>>
    >>> rpropnet = algorithms.IRPROPPlus(
    ...     (2, 3, 1),
    ...     verbose=False
    ... )
    >>> rpropnet.train(x_train, y_train)

    See Also
    --------
    :network:`RPROP` : RPROP algorithm.
    :network:`GradientDescent` : GradientDescent algorithm.
    """

    def init_variables(self):
        super(IRPROPPlus, self).init_variables()
        self.variables.update(
            last_error=theano.shared(name='last_error', value=np.nan),
            previous_error=theano.shared(name='previous_error', value=np.nan),
        )

    def on_epoch_start_update(self, epoch):
        super(IRPROPPlus, self).on_epoch_start_update(epoch)

        previous_error = self.errors.previous()
        if previous_error:
            last_error = self.errors.last()
            self.variables.last_error.set_value(last_error)
            self.variables.previous_error.set_value(previous_error)

    def init_prev_delta(self, parameter):
        last_error = self.variables.last_error
        prev_error = self.variables.previous_error
        return T.switch(last_error > prev_error, parameter.prev_delta, 0)
