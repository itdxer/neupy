import theano
from theano.ifelse import ifelse
import numpy as np

from neupy.core.properties import (BoundedProperty,
                                   BetweenZeroAndOneProperty)
from .base import LearningRateConfigurable


__all__ = ('ErrorDifferenceStepUpdate',)


class ErrorDifferenceStepUpdate(LearningRateConfigurable):
    """ This algorithm make step update base on error difference between
    epochs.

    Parameters
    ----------
    update_for_smaller_error : float
        Multiplies this option to ``step`` in if the error was less than in
        previous epochs. Defaults to ``1.05``. Value can't be less
        than ``1``.
    update_for_bigger_error : float
        Multiplies this option to ``step`` in if the error was more than in
        previous epochs. Defaults to ``0.7``.
    error_difference : float
        The value indicates how many had to increase the error from the
        previous epochs that would produce a reduction step. Defaults
        to ``1.04``. Value can't be less than ``1``.

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
    ...     optimizations=[algorithms.ErrorDifferenceStepUpdate]
    ... )
    >>>
    """
    update_for_smaller_error = BoundedProperty(default=1.05, min_size=1)
    update_for_bigger_error = BetweenZeroAndOneProperty(default=0.7)
    error_difference = BoundedProperty(default=1.04, min_size=1)

    def init_variables(self):
        self.variables.update(
            last_error=theano.shared(name='last_error', value=np.nan),
            previous_error=theano.shared(name='previous_error', value=np.nan),
        )
        super(ErrorDifferenceStepUpdate, self).init_variables()

    def init_train_updates(self):
        updates = super(ErrorDifferenceStepUpdate, self).init_train_updates()

        step = self.variables.step
        last_error = self.variables.last_error
        previous_error = self.variables.previous_error

        step_update_condition = ifelse(
            last_error < previous_error,
            self.update_for_smaller_error * step,
            ifelse(
                last_error > self.update_for_bigger_error * previous_error,
                self.update_for_bigger_error * step,
                step
            )

        )
        updates.append((step, step_update_condition))
        return updates

    def epoch_start_update(self, epoch):
        super(ErrorDifferenceStepUpdate, self).epoch_start_update(epoch)

        previous_error = self.previous_error()
        if previous_error:
            last_error = self.last_error()
            self.variables.last_error.set_value(last_error)
            self.variables.previous_error.set_value(previous_error)
