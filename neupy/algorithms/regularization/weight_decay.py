from neupy.utils import asfloat
from neupy.layers.utils import iter_parameters
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

    def init_train_updates(self):
        original_updates = super(WeightDecay, self).init_train_updates()
        parameters = [param for _, _, param in iter_parameters(self.layers)]
        modified_updates = []

        step = self.variables.step
        decay_rate = asfloat(self.decay_rate)

        for parameter, updated in original_updates:
            if parameter in parameters:
                updated -= step * decay_rate * parameter
            modified_updates.append((parameter, updated))

        return modified_updates
