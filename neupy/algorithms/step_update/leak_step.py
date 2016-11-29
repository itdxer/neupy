import numpy as np
import theano
import theano.tensor as T

from neupy.utils import asfloat
from neupy.core.properties import ProperFractionProperty, BoundedProperty
from neupy.algorithms.utils import parameter_values
from neupy.layers.utils import count_parameters
from .base import SingleStepConfigurable


__all__ = ('LeakStepAdaptation',)


class LeakStepAdaptation(SingleStepConfigurable):
    """
    Leak Learning Rate Adaptation algorithm is a step
    adaptation procedure in backpropagation algortihm.

    Parameters
    ----------
    leak_size : float
        Defaults to ``0.01``. This variable identified
        proportion, so it's always between 0 and 1.
        Typically this value is small.

    alpha : float
        The ``alpha`` is control total step update ratio.
        Defaults to ``0.001``. Typically this value is small.

    beta : float
        This similar to ``alpha``, but it control ration
        only for update matrix norms. Defaults to ``20``.
        Typically this value is bigger than ``1``.

    Warns
    -----
    {SingleStepConfigurable.Warns}

    Examples
    --------
    >>> from neupy import algorithms
    >>> bpnet = algorithms.GradientDescent(
    ...     (2, 4, 1),
    ...     addons=[algorithms.LeakStepAdaptation]
    ... )

    References
    ----------
    [1] Noboru M. "Adaptive on-line learning in changing
        environments", 1997

    [2] LeCun, "Efficient BackProp", 1998
    """
    leak_size = ProperFractionProperty(default=0.01)
    alpha = BoundedProperty(default=0.001, minval=0)
    beta = BoundedProperty(default=20, minval=0)

    def init_variables(self):
        super(LeakStepAdaptation, self).init_variables()
        n_parameters = count_parameters(self.connection)
        self.variables.leak_average = theano.shared(
            name='leak-step-adapt/leak-average',
            value=asfloat(np.zeros(n_parameters)),
        )

    def init_train_updates(self):
        updates = super(LeakStepAdaptation, self).init_train_updates()

        alpha = self.alpha
        beta = self.beta
        leak_size = self.leak_size

        step = self.variables.step
        leak_average = self.variables.leak_average

        parameters = parameter_values(self.connection)
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
