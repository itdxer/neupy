from neupy.core.properties import BoundedProperty
from .base import WeightUpdateConfigurable


__all__ = ('WeightDecay',)


class WeightDecay(WeightUpdateConfigurable):
    """
    Weight decay algorithm penalizes large weights and
    limits the freedom in network. The algorithm is able
    to solve one of the possible problems of network's
    overfitting.

    Parameters
    ----------
    decay_rate : float
        Controls the effect of penalties on the update
        network weights. Defaults to ``0.1``.

    Warns
    -----
    {WeightUpdateConfigurable.Warns}

    Examples
    --------
    >>> from neupy import algorithms
    >>> bpnet = algorithms.GradientDescent(
    ...     (2, 4, 1),
    ...     step=0.1,
    ...     decay_rate=0.1,
    ...     addons=[algorithms.WeightDecay]
    ... )

    See Also
    --------
    :network:`WeightElimination`
    """
    decay_rate = BoundedProperty(default=0.1, minval=0)

    def init_param_updates(self, layer, parameter):
        updates = super(WeightDecay, self).init_param_updates(
            layer, parameter
        )
        step = self.variables.step
        updates_mapper = dict(updates)
        updates_mapper[parameter] -= step * self.decay_rate * parameter
        return list(updates_mapper.items())
