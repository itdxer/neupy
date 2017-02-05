from neupy.core.properties import BoundedProperty
from neupy.utils import asfloat
from .base import WeightUpdateConfigurable


__all__ = ('WeightElimination',)


class WeightElimination(WeightUpdateConfigurable):
    """
    Weight Elimination algorithm penalizes large weights
    and limits the freedom in network. The algorithm is
    able to solve one of the possible problems of network
    overfitting.

    Parameters
    ----------
    decay_rate : float
        Controls the effect of penalties on the update
        network weights. Defaults to ``0.1``.

    zero_weight : float
        Second important parameter for weights penalization.
        Defaults to ``1``. Small value can make all weights
        close to zero. Big value will make less significant
        contribution in weights update. Which mean that with
        a bigger value of the ``zero_weight`` parameter network
        allows higher values for the weights.

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
    ...     addons=[algorithms.WeightElimination]
    ... )

    See Also
    --------
    :network:`WeightDecay` : Weight Decay penalty.

    Notes
    -----
    Before adding that regularization parameter carefully
    choose ``decay_rate`` and ``zero_weight`` parameters
    for the problem. Invalid parameters can make weight
    very close to the origin (all values become
    close to zero).

    References
    ----------
    [1] Weigend, A. S.; Rumelhart, D. E. & Huberman, B. A. (1991),
        Generalization by Weight-Elimination with Application to
        Forecasting, in Richard P. Lippmann; John E. Moody & David S.
        Touretzky, ed., Advances in Neural Information Processing
        Systems, San Francisco, CA: Morgan Kaufmann, pp. 875--882 .
    """
    decay_rate = BoundedProperty(default=0.1, minval=0)
    zero_weight = BoundedProperty(default=1, minval=0)

    def init_param_updates(self, layer, parameter):
        updates = super(WeightElimination, self).init_param_updates(
            layer, parameter
        )

        step = self.variables.step
        decay_koef = asfloat(self.decay_rate * step)
        zero_weight_square = asfloat(self.zero_weight ** 2)

        updates_mapper = dict(updates)
        updates_mapper[parameter] -= decay_koef * (
            (2 * parameter / zero_weight_square) / (
                1 + (parameter ** 2) / zero_weight_square
            ) ** 2
        )

        return list(updates_mapper.items())
