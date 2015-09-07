from neupy.core.properties import (NonNegativeNumberProperty,
                                   BetweenZeroAndOneProperty)
from .base import SingleStep


__all__ = ('ErrorDifferenceStepUpdate',)


class ErrorDifferenceStepUpdate(SingleStep):
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
    update_for_smaller_error = NonNegativeNumberProperty(default=1.05,
                                                         min_size=1)
    update_for_bigger_error = BetweenZeroAndOneProperty(default=0.7)
    error_difference = NonNegativeNumberProperty(default=1.04, min_size=1)

    def new_step(self):
        current_step = self.step

        if not self.errors_in:
            return current_step

        last_error = self.last_error_in()
        previous_error = self.previous_error()

        if previous_error is None:
            return current_step

        elif last_error < previous_error:
            return self.update_for_smaller_error * current_step

        elif last_error >= self.error_difference * previous_error:
            return self.update_for_bigger_error * current_step

        return current_step

    def after_weight_update(self, input_train, target_train):
        super(ErrorDifferenceStepUpdate, self).after_weight_update(
            input_train, target_train
        )
        self.step = self.new_step()
