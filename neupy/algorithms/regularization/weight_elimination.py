import tensorflow as tf

from neupy.core.properties import BoundedProperty
from neupy.utils import asfloat
from neupy.layers.utils import iter_parameters
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

    def init_train_updates(self):
        original_updates = super(WeightElimination, self).init_train_updates()
        parameters = [param for _, _, param in iter_parameters(self.layers)]
        modified_updates = []

        step = self.variables.step
        decay_koef = asfloat(self.decay_rate * step)
        zero_weight_square = asfloat(self.zero_weight ** 2)

        for parameter, updated in original_updates:
            if parameter in parameters:
                updated -= decay_koef * (
                    (2 * parameter / zero_weight_square) / tf.square(
                        1 + tf.square(parameter) / zero_weight_square
                    )
                )
            modified_updates.append((parameter, updated))

        return modified_updates
