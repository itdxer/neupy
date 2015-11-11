from neupy.core.properties import NonNegativeNumberProperty
from .base import WeightUpdateConfigurable


__all__ = ('WeightElimination',)


class WeightElimination(WeightUpdateConfigurable):
    """ Weight Elimination algorithm penalizes large weights and limits the
    freedom in network. The algorithm is able to solve one of the possible
    problems of network overfitting.

    Parameters
    ----------
    decay_rate : float
        Controls the effect of penalties on the update network weights.
        Defaults to ``0.1``.
    zero_weight : float
        Second important parameter for weights penalization. Defaults
        to ``1``. Small value can make all weights close to zero. Big value
        will make less significant contribution in weight update. That mean
        with a big value ``zero_weight`` network allow higher values for
        the weights.

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
    ...     optimizations=[algorithms.WeightElimination]
    ... )
    >>>

    See Also
    --------
    :network:`WeightDecay`
    """
    decay_rate = NonNegativeNumberProperty(default=0.1)
    zero_weight = NonNegativeNumberProperty(default=1)

    def init_layer_update(self, layer):
        updates = super(WeightElimination, self).init_layer_update(layer)
        modified_updates = []

        step = layer.step or self.variables.step
        decay_koef = self.decay_rate * step
        zero_weight_square = self.zero_weight ** 2

        for update_var, update_func in updates:
            if update_var.name.startswith(('weight', 'bias')):
                update_func -= decay_koef * (
                    (2 * update_var / zero_weight_square) / (
                        1 + (update_var ** 2) / zero_weight_square
                    ) ** 2
                )
            modified_updates.append((update_var, update_func))
        return modified_updates
