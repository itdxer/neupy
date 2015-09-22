import copy

from numpy import where, sign, ones, clip

from neupy.core.properties import (NonNegativeNumberProperty,
                                   BetweenZeroAndOneProperty)
from .backpropagation import Backpropagation


__all__ = ('RPROP', 'IRPROPPlus')


class RPROP(Backpropagation):
    """ RPROP :network:`Backpropagation` algorithm optimization.

    Parameters
    ----------
    {rprop_params}
    {optimizations}
    {raw_predict_param}
    {full_params}

    Methods
    -------
    {supervised_train}
    {full_methods}

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
    :network:`Backpropagation` : Backpropagation algorithm.
    """

    __rprop_params = """minimum_step : float
        Minimum possible value for step. Defaults to ``0.1``.
    maximum_step : float
        Maximum possible value for step. Defaults to ``50``.
    increase_factor : float
        Increase factor for step in case when gradient doesn't change
        sign compare to previous epoch.
    decrease_factor : float
        Decrease factor for step in case when gradient changes sign
        compare to previous epoch.
    """

    shared_docs = {"rprop_params": __rprop_params}

    # This properties correct upper and lower bounds for steps.
    minimum_step = NonNegativeNumberProperty(default=0.1)
    maximum_step = NonNegativeNumberProperty(default=50)

    # This properties increase/decrease step by deviding it to
    # some coeffitient.
    increase_factor = NonNegativeNumberProperty(min_size=1, default=1.2)
    decrease_factor = BetweenZeroAndOneProperty(default=0.5)

    def init_layers(self):
        super(RPROP, self).init_layers()
        steps = self.steps = []

        for layer in self.train_layers:
            steps.append(ones(layer.size) * self.step)

    def get_flip_sign_weight_delta(self, layer_number):
        return self.prev_weight_deltas[layer_number]

    def layer_weight_update(self, delta, layer_number):
        if not hasattr(self, 'prev_gradients'):
            prev_gradient = 0
            prev_weight_delta = 0
        else:
            prev_gradient = self.prev_gradients[layer_number]
            prev_weight_delta = self.get_flip_sign_weight_delta(layer_number)

        step = self.steps[layer_number]
        gradient = self.gradients[layer_number]

        grad_product = prev_gradient * gradient
        negative_gradients = grad_product < 0

        step = self.steps[layer_number] = clip(
            where(
                grad_product > 0,
                # Increase step for gradients which switch signs
                step * self.increase_factor,
                where(
                    negative_gradients,
                    # Decrease step for gradients whcih switch signs
                    step * self.decrease_factor,
                    # Setup the same step value
                    step
                )
            ),
            self.minimum_step,
            self.maximum_step,
        )

        output = where(
            negative_gradients,
            -prev_weight_delta,
            -sign(gradient) * step
        )
        gradient[negative_gradients] = 0

        self.weight_deltas.append(output.copy())

        return output

    def update_weights(self, weight_deltas):
        self.weight_deltas = []
        super(RPROP, self).update_weights(weight_deltas)

        self.prev_weight_deltas = copy.copy(self.weight_deltas)
        self.prev_gradients = copy.copy(self.gradients)
        self.prev_steps = copy.copy(self.steps)


class IRPROPPlus(RPROP):
    """ iRPROP+ :network:`Backpropagation` algorithm optimization.

    Parameters
    ----------
    {rprop_params}
    {optimizations}
    {full_params}

    Methods
    -------
    {supervised_train}
    {full_methods}

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
    :network:`Backpropagation` : Backpropagation algorithm.
    """
    def get_flip_sign_weight_delta(self, layer_number):
        prev_error = self.previous_error()
        last_error = self.last_error()
        prev_weight_delta = 0

        if prev_error is None or last_error > prev_error:
            prev_weight_delta = self.prev_weight_deltas[layer_number]

        return prev_weight_delta
