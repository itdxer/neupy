import copy

from neupy.core.properties import BetweenZeroAndOneProperty
from .backpropagation import Backpropagation


__all__ = ('Momentum',)


class Momentum(Backpropagation):
    """ Momentum algorithm for :network:`Backpropagation` optimization.

    Parameters
    ----------
    momentum : float
        Control previous gradient ratio. Defaults to ``0.9``.
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
    >>> mnet = algorithms.Momentum(
    ...     (2, 3, 1),
    ...     verbose=False
    ... )
    >>> mnet.train(x_train, y_train)

    See Also
    --------
    :network:`Backpropagation` : Backpropagation algorithm.
    """
    momentum = BetweenZeroAndOneProperty(default=0.9)

    def layer_weight_update(self, delta, layer_number):
        update = super(Momentum, self).layer_weight_update(delta,
                                                           layer_number)
        if not hasattr(self, 'prev_gradients'):
            return update
        return -self.momentum * self.prev_gradients[layer_number] + update

    def update_weights(self, weight_deltas):
        super(Momentum, self).update_weights(weight_deltas)
        self.prev_gradients = copy.copy(self.gradients)
