from neupy.core.properties import IntProperty
from .base import SingleStepConfigurable


__all__ = ('StepDecay',)


class StepDecay(SingleStepConfigurable):
    """
    Algorithm minimizes learning step monotonically after
    each iteration.

    .. math::

        \\alpha_{{t + 1}} = \\frac{{\\alpha_{{0}}}}\
            {{1 + \\frac{{t}}{{m}}}}

    where :math:`\\alpha` is a step, :math:`t` is an epoch number
    and :math:`m` is a ``reduction_freq`` parameter.

    Parameters
    ----------
    reduction_freq : int
        Parameter controls step redution frequency.
        The higher the value the slower step parameter
        decreases.

        For instance, if ``reduction_freq=100``
        and ``step=0.12`` then after ``100`` epochs ``step`` is
        going to be equal to ``0.06`` (which is ``0.12 / 2``),
        after ``200`` epochs ``step`` is going to be equal to
        ``0.04`` (which is ``0.12 / 3``) and so on.

        Defaults to ``100`` epochs.

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
    ...     reduction_freq=100,
    ...     addons=[algorithms.StepDecay]
    ... )
    >>>

    See Also
    --------
    :network:`SearchThenConverge`
    """
    reduction_freq = IntProperty(minval=1, default=100)

    def init_train_updates(self):
        updates = super(StepDecay, self).init_train_updates()
        epoch = self.variables.epoch
        step = self.variables.step

        step_update_condition = self.step / (
            1 + epoch / self.reduction_freq
        )
        updates.extend([
            (step, step_update_condition),
        ])
        return updates
