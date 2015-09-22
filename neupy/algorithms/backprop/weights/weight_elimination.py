from neupy.core.properties import NonNegativeNumberProperty
from .base import WeightUpdateConfigurable


__all__ = ('WeightElimination',)


class WeightElimination(WeightUpdateConfigurable):
    """ Weight elimination algorithm penalizes large weights and limits the
    freedom in network. The algorithm is able to solve one of the possible
    problems of network overfitting.

    Parameters
    ----------
    decay_rate : float
        Controls the effect of penalties on the update network weights.
        Defaults to ``0.1``.
    zero_weight : float
        Second important parameter for weights penalization. Defaults
        to ``1``.

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

    def layer_weight_update(self, delta, layer_number):
        weight_update = super(WeightElimination, self).layer_weight_update(
            delta, layer_number
        )

        weight = self.train_layers[layer_number].weight
        step = self.layer_step(layer_number)
        decay_koef = self.decay_rate * step
        zero_weight_square = self.zero_weight ** 2

        return weight_update + decay_koef * (
            (2 * weight / zero_weight_square) / (
                1 + (weight ** 2) / zero_weight_square
            ) ** 2
        )
