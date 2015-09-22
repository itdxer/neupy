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

    def layer_weight_update(self, delta, layer_number):
        weight_update = super(WeightDecay, self).layer_weight_update(
            delta, layer_number
        )

        weight = self.train_layers[layer_number].weight
        step = self.layer_step(layer_number)

        return -step * self.decay_rate * weight + weight_update
