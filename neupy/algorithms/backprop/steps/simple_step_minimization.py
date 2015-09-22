from neupy.core.properties import NonNegativeIntProperty
from .base import SingleStep


__all__ = ('SimpleStepMinimization',)


class SimpleStepMinimization(SingleStep):
    """ Algorithm Monotonicly minimize learning step on each iteration.
    Probably this is most simple step minimization idea.

    Parameters
    ----------
    epochs_step_minimizator : int
        The parameter controls the frequency reduction step with respect
        to epochs. Defaults to ``100`` epochs. Can't be less than ``1``.
        Less value mean that step decrease faster.

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
    ...     optimizations=[algorithms.SimpleStepMinimization]
    ... )
    >>>

    See Also
    --------
    :network:`SearchThenConverge`
    """
    epochs_step_minimizator = NonNegativeIntProperty(min_size=1, default=100)

    def after_weight_update(self, input_train, target_train):
        super(SimpleStepMinimization, self).after_weight_update(
            input_train, target_train
        )
        self.step = self.first_step / (
            1 + self.epoch / self.epochs_step_minimizator
        )
