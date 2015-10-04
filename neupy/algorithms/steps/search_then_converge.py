from __future__ import division

from neupy.core.properties import NonNegativeIntProperty, NumberProperty
from .base import SingleStep


__all__ = ('SearchThenConverge',)


class SearchThenConverge(SingleStep):
    """ Algorithm minimize learning step. Similar to
    :network:`SimpleStepMinimization`, but more complicated step update rule.

    Parameters
    ----------
    epochs_step_minimizator : int
        The parameter controls the frequency reduction step with respect
        to epochs. Defaults to ``100`` epochs. Can't be less than ``1``.
        Less value mean that step decrease faster.
    rate_coefitient : float
        Second important parameter to control the rate of error reduction.
        Defaults to ``0.2``

    Attributes
    ----------
    {first_step}

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
    ...     optimizations=[algorithms.SearchThenConverge]
    ... )
    >>>

    See Also
    --------
    :network:`SimpleStepMinimization`
    """
    epochs_step_minimizator = NonNegativeIntProperty(min_size=1, default=100)
    rate_coefitient = NumberProperty(default=0.2)

    def after_weight_update(self, input_train, target_train):
        super(SearchThenConverge, self).after_weight_update(
            input_train, target_train
        )

        first_step = self.first_step
        epochs_step_minimizator = self.epochs_step_minimizator

        epoch_value = self.epoch / epochs_step_minimizator
        rated_value = (self.rate_coefitient / first_step) * epoch_value

        self.step = first_step * (1 + rated_value) / (
            1 + rated_value + epochs_step_minimizator * epoch_value ** 2
        )
