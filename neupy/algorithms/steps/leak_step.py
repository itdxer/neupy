import numpy as np
import theano
import theano.tensor as T

from neupy.utils import asfloat
from neupy.core.properties import ProperFractionProperty, BoundedProperty
from .base import MultipleStepConfigurable


__all__ = ('LeakStepAdaptation',)


class LeakStepAdaptation(MultipleStepConfigurable):
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

    Warns
    -----
    {MultipleStepConfigurable.Warns}

    Examples
    --------
    >>> from neupy import algorithms
    >>>
    >>> bpnet = algorithms.GradientDescent(
    ...     (2, 4, 1),
    ...     step=0.1,
    ...     verbose=False,
    ...     addons=[algorithms.LeakStepAdaptation]
    ... )
    >>>
    """
    leak_size = ProperFractionProperty(default=0.5)
    alpha = BoundedProperty(default=0.5, minval=0)
    beta = BoundedProperty(default=0.5, minval=0)

    def init_layers(self):
        super(LeakStepAdaptation, self).init_layers()
        for layer in self.train_layers:
            layer.leak_avarage = theano.shared(
                value=asfloat(np.zeros(layer.weight_shape)),
                name='layer_leak_avarage'
            )
            layer.step = theano.shared(value=self.step, name='layer_step')

    def init_layer_updates(self, layer):
        updates = super(LeakStepAdaptation, self).init_layer_updates(layer)

        alpha = self.alpha
        beta = self.beta
        leak_size = self.leak_size

        grad_w = T.grad(self.variables.error_func, wrt=layer.weight)
        step = layer.step
        leak_average = layer.leak_avarage

        leak_avarage_update = (
            (1 - leak_size) * leak_average + leak_size * grad_w
        )
        updates.extend([
            (leak_average, leak_avarage_update),
            (
                step,
                step + alpha * step * (
                    beta * leak_avarage_update.norm(2) - step
                )
            ),
        ])

        return updates
