from neupy.core.properties import NonNegativeNumberProperty
from .base import WeightUpdateConfigurable


__all__ = ('WeightDecay',)


class WeightDecay(WeightUpdateConfigurable):
    """ Weight decay algorithm penalizes large weights and limits the
    freedom in network. The algorithm is able to solve one of the possible
    problems of network overfitting.

    Parameters
    ----------
    decay_rate : float
        Controls the effect of penalties on the update network weights.
        Defaults to ``0.1``.

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
    ...     optimizations=[algorithms.WeightDecay]
    ... )
    >>>

    See Also
    --------
    :network:`WeightElimination`
    """
    decay_rate = NonNegativeNumberProperty(default=0.1)

    def init_layer_update(self, layer):
        updates = super(WeightDecay, self).init_layer_update(layer)
        modified_updates = []
        step = layer.step or self.variables.step

        for update_var, update_func in updates:
            if update_var.name.startswith(('weight', 'bias')):
                update_func -= step * self.decay_rate * update_var
            modified_updates.append((update_var, update_func))
        return modified_updates
