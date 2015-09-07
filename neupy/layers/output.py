from numpy import zeros, round as np_round, where

from neupy.network.connections import NetworkConnectionError
from neupy.layers.base import BaseLayer
from neupy.core.properties import (NumberBoundProperty, NonNegativeIntProperty,
                                   BetweenZeroAndOneProperty)


__all__ = ('OutputLayer', 'CompetitiveOutputLayer', 'StepOutputLayer',
           'RoundOutputLayer')


class OutputLayer(BaseLayer):
    """ Simples output layer class which does not make any transformations.
    Output from this layer is the same as input.

    Parameters
    ----------
    {layer_params}
    """
    def initialize(self, *args, **kwargs):
        return

    def relate_to(self, right_layer):
        raise NetworkConnectionError("Can't connect from output layer")

    def format_output(self, value):
        return value

    def output(self, value):
        return self.format_output(value)


class CompetitiveOutputLayer(OutputLayer):
    """ Competitive layer output. Layer output will return the result where
    all zero values and one value which has greatest value will be one.

    Parameters
    ----------
    {layer_params}
    """
    def format_output(self, value):
        output = zeros(value.shape)
        max_args = value.argmax(axis=1)
        output[range(value.shape[0]), max_args] = 1
        return output


class StepOutputLayer(OutputLayer):
    """ The behaviour for this layer is the same as for step function.

    Parameters
    ----------
    output_bounds : tuple
        Value is must be a tuple which contains two elements where first one
        identify lower output value and the second one - bigger. Defaults
        to ``(0, 1)``.
    critical_point : float
        Critical point is setup step function bias.
    {layer_params}
    """
    output_bounds = NumberBoundProperty(default=(0, 1))
    critical_point = BetweenZeroAndOneProperty(default=0.5)

    def format_output(self, value):
        lower_bound, upper_bound = self.output_bounds
        return where(value < self.critical_point, lower_bound, upper_bound)


class RoundOutputLayer(OutputLayer):
    """ Round output layer value.

    Parameters
    ----------
    decimal_places : int
        The precision in decimal digits for output value.
    {layer_params}
    """
    decimal_places = NonNegativeIntProperty(default=0)

    def format_output(self, value):
        return np_round(value, self.decimal_places)
