from __future__ import division

from neupy.core.properties import IntProperty, NumberProperty
from .base import SingleStepConfigurable


__all__ = ('SearchThenConverge',)


class SearchThenConverge(SingleStepConfigurable):
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

    Warns
    -----
    {SingleStepConfigurable.Warns}

    Examples
    --------
    >>> from neupy import algorithms
    >>>
    >>> bpnet = algorithms.GradientDescent(
    ...     (2, 4, 1),
    ...     step=0.1,
    ...     verbose=False,
    ...     addons=[algorithms.SearchThenConverge]
    ... )
    >>>

    See Also
    --------
    :network:`SimpleStepMinimization`
    """
    epochs_step_minimizator = IntProperty(minval=1, default=100)
    rate_coefitient = NumberProperty(default=0.2)

    def init_train_updates(self):
        updates = super(SearchThenConverge, self).init_train_updates()

        first_step = self.step
        epochs_step_minimizator = self.epochs_step_minimizator

        step = self.variables.step
        epoch = self.variables.epoch

        epoch_value = epoch / epochs_step_minimizator
        rated_value = 1 + (self.rate_coefitient / first_step) * epoch_value
        step_update_condition = (first_step * rated_value) / (
            rated_value + epochs_step_minimizator * epoch_value ** 2
        )

        updates.append((step, step_update_condition))
        return updates
