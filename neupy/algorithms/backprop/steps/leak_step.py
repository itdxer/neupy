from numpy import zeros
from numpy.linalg import norm

from neupy.core.properties import (BetweenZeroAndOneProperty,
                                   NonNegativeNumberProperty)
from .base import MultiSteps


__all__ = ('LeakStepAdaptation',)


class LeakStepAdaptation(MultiSteps):
    """ Leak Learning Rate Adaptation algorithm for step adaptation procedure
    in backpropagation algortihm. By default every layer has the same value
    as ``step`` parameter in network, but after first training epoch they
    must be different.

    Parameters
    ----------
    leak_size : float
        Leak size control ratio of update variable which combine weight
        deltas from previous epochs, defaults to ``0.5``.
    alpha : float
        The ``alpha`` is control total step update ratio (It's similar to
        step role in weight update procedure). Defaults to ``0.5``.
    beta : float
        This similar to ``alpha``, but it control ration only for update
        matrix norms. Defaults to ``0.5``.

    Attributes
    ----------
    {steps}

    Warns
    -----
    {bp_depending}

    Examples
    --------
    >>> from neupy import algorithms
    >>>
    >>> bpnet = algorithms.Backpropagation(
    ...     (2, 4, 1),
    ...     step=0.1,
    ...     verbose=False,
    ...     optimizations=[algorithms.LeakStepAdaptation]
    ... )
    >>>
    """
    leak_size = BetweenZeroAndOneProperty(default=0.5)
    alpha = NonNegativeNumberProperty(default=0.5)
    beta = NonNegativeNumberProperty(default=0.5)

    def init_layers(self):
        super(LeakStepAdaptation, self).init_layers()
        updates = self.updates = []

        for layer in self.train_layers:
            updates.append(zeros(layer.size))

    def after_weight_update(self, input_train, target_train):
        super(LeakStepAdaptation, self).after_weight_update(input_train,
                                                            target_train)
        alpha = self.alpha
        beta = self.beta
        leak_size = self.leak_size

        weight_delta = self.weight_delta
        steps = self.steps
        updates = self.updates

        for i, layer in enumerate(self.train_layers):
            step = steps[i]
            update = updates[i]

            updates[i] = (1 - leak_size) * update + (
                leak_size * weight_delta[i]
            )
            steps[i] += alpha * step * (beta * norm(updates[i]) - step)
