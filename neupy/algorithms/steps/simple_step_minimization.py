from neupy.core.properties import IntProperty
from .base import SingleStepConfigurable


__all__ = ('SimpleStepMinimization',)


class SimpleStepMinimization(SingleStepConfigurable):
    """ Algorithm Monotonicly minimize learning step on each iteration.
    Probably this is most simple step minimization idea.

    Parameters
    ----------
    epochs_step_minimizator : int
        The parameter controls the frequency reduction step with respect
        to epochs. Defaults to ``100`` epochs. Can't be less than ``1``.
        Less value mean that step decrease faster.

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
    ...     addons=[algorithms.SimpleStepMinimization]
    ... )
    >>>

    See Also
    --------
    :network:`SearchThenConverge`
    """
    epochs_step_minimizator = IntProperty(minval=1, default=100)

    def init_train_updates(self):
        updates = super(SimpleStepMinimization, self).init_train_updates()
        variables = self.variables
        step_update_condition = self.step / (
            1 + variables.epoch / self.epochs_step_minimizator
        )
        updates.append((variables.step, step_update_condition))
        return updates
