import numpy as np

from neupy.network.connections import NetworkConnectionError
from neupy.layers.base import BaseLayer
from neupy.core.properties import (IntBoundProperty, NonNegativeIntProperty,
                                   BetweenZeroAndOneProperty)


__all__ = ('Output', 'CompetitiveOutput', 'StepOutput', 'RoundedOutput',
           'ArgmaxOutput')


class Output(BaseLayer):
    """ Simples output layer class which does not make any transformations.
    Output from this layer is the same as input.

    Parameters
    ----------
    {input_size_param}
    """

    def relate_to(self, right_layer):
        raise NetworkConnectionError("Can't create connection "
                                     "from output layer")

    def output(self, value):
        return value


class CompetitiveOutput(Output):
    """ Competitive layer output. Layer output will return the result where
    all zero values and one value which has greatest value will be one.

    Parameters
    ----------
    {input_size_param}
    """
    def output(self, value):
        output = np.zeros(value.shape, dtype=np.int0)
        max_args = value.argmax(axis=1)
        output[range(value.shape[0]), max_args] = 1
        return output


class StepOutput(Output):
    """ The behaviour for this layer is the same as for step function.

    Parameters
    ----------
    output_bounds : tuple
        Value is must be a tuple which contains two elements where first one
        identify lower output value and the second one - bigger. Defaults
        to ``(0, 1)``.
    critical_point : float
        Critical point is set up step function bias. Value equal to this
        point should be equal to the lower bound. Defaults to ``0``.
    {input_size_param}
    """
    output_bounds = IntBoundProperty(default=(0, 1))
    critical_point = BetweenZeroAndOneProperty(default=0)

    def output(self, value):
        lower_bound, upper_bound = self.output_bounds
        return np.where(value <= self.critical_point,
                        lower_bound, upper_bound)


class RoundedOutput(Output):
    """ Round output layer value.

    Parameters
    ----------
    decimal_places : int
        The precision in decimal digits for output value.
    {input_size_param}
    """
    decimal_places = NonNegativeIntProperty(default=0)

    def output(self, value):
        return np.round(value, self.decimal_places)


class ArgmaxOutput(Output):
    """ Return number of feature that have maximum value for each sample.

    Parameters
    ----------
    decimal_places : int
    The precision in decimal digits for output value.
    {input_size_param}
    """

    def output(self, value):
        return value.argmax(axis=1)
