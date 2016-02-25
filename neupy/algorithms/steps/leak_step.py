import numpy as np
import theano
import theano.tensor as T

from neupy.utils import asfloat
from neupy.core.properties import ProperFractionProperty, BoundedProperty
from neupy.algorithms.utils import iter_parameters, count_parameters
from .base import SingleStepConfigurable


__all__ = ('LeakStepAdaptation',)


class LeakStepAdaptation(SingleStepConfigurable):
    """ Leak Learning Rate Adaptation algorithm for step adaptation procedure
    in backpropagation algortihm. By default every layer has the same value
    as ``step`` parameter in network, but after first training epoch they
    must be different.

    Parameters
    ----------
    leak_size : float
        Leak size control ratio of update variable which combine weight
        deltas from previous epochs, defaults to ``0.5``.
    alpha : float
        The ``alpha`` is control total step update ratio (It's similar to
        step role in weight update procedure). Defaults to ``0.5``.
    beta : float
        This similar to ``alpha``, but it control ration only for update
        matrix norms. Defaults to ``0.5``.

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
    ...     addons=[algorithms.LeakStepAdaptation]
    ... )
    >>>
    """
    leak_size = ProperFractionProperty(default=0.5)
    alpha = BoundedProperty(default=0.5, minval=0)
    beta = BoundedProperty(default=0.5, minval=0)

    def init_variables(self):
        super(LeakStepAdaptation, self).init_variables()
        n_parameters = count_parameters(self)
        self.variables.leak_average = theano.shared(
            value=asfloat(np.zeros(n_parameters)),
            name='leak_average'
        )

    def init_train_updates(self):
        updates = super(LeakStepAdaptation, self).init_train_updates()

        alpha = self.alpha
        beta = self.beta
        leak_size = self.leak_size

        step = self.variables.step
        leak_average = self.variables.leak_average

        parameters = list(iter_parameters(self))
        gradients = T.grad(self.variables.error_func, wrt=parameters)
        full_gradient = T.concatenate([grad.flatten() for grad in gradients])

        leak_avarage_update = (
            (1 - leak_size) * leak_average + leak_size * full_gradient
        )
        new_step = step + alpha * step * (
            beta * leak_avarage_update.norm(L=2) - step
        )

        updates.extend([
            (leak_average, leak_avarage_update),
            (step, new_step),
        ])

        return updates
