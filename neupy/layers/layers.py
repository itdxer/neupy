from numpy import arccos, dot, reshape
from numpy.linalg import norm

from neupy.core.properties import DictProperty
from neupy.functions import get_partial_for_func
from neupy.functions import (linear, sigmoid, step, tanh, rectifier,
                             softplus, softmax)
from neupy.layers.base import BaseLayer


__all__ = ('Layer', 'LinearLayer', 'SigmoidLayer', 'StepLayer', 'TanhLayer',
           'RectifierLayer', 'SoftplusLayer', 'SoftmaxLayer',
           'EuclideDistanceLayer', 'AngleDistanceLayer')


class Layer(BaseLayer):
    """ Base class for neural network layers.

    Parameters
    ----------
    function_coef : dict
        Default settings for activation function.
    {layer_params}
    """

    function_coef = DictProperty()

    def __init__(self, *args, **kwargs):
        super(Layer, self).__init__(*args, **kwargs)

        if self.function_coef is not None:
            partial_func = get_partial_for_func(self.activation_function)
            self.activation_function = partial_func(
                self.activation_function, **self.function_coef
            )


class LinearLayer(Layer):
    """ Layer with linear activation function.

    Parameters
    ----------
    {layer_params}
    """
    activation_function = linear


class SigmoidLayer(Layer):
    """ Layer with sigmoid activation function.

    Parameters
    ----------
    function_coef : dict
        Default configurations for signoid activation function. There is one
        value name ``alpha`` (default is ``1``). `alpha` control your
        function shape.
    {layer_params}
    """
    function_coef = DictProperty(default={'alpha': 1})
    activation_function = sigmoid


class StepLayer(Layer):
    """ Layer with step activation function.

    Parameters
    ----------
    {layer_params}
    """
    activation_function = step


class TanhLayer(Layer):
    """ Layer with `tanh` activation function.

    Parameters
    ----------
    function_coef : dict
        Default configurations for sigmoid activation function. There is one
        value name ``alpha`` (default is ``1``). `alpha` control your
        function shape.
    {layer_params}
    """
    function_coef = DictProperty(default={'alpha': 1})
    activation_function = tanh


class RectifierLayer(Layer):
    """ Layer with rectifier activation function.

    Parameters
    ----------
    {layer_params}
    """
    activation_function = rectifier


class SoftplusLayer(Layer):
    """ Layer with softplus activation function.

    Parameters
    ----------
    {layer_params}
    """
    activation_function = softplus


class SoftmaxLayer(Layer):
    """ Layer with softmax activation function.

    Parameters
    ----------
    function_coef : dict
        Default configurations for softmax activation function. There is one
        value name ``temp`` (default is ``1``). Smaller ``temp`` value will
        make your winner probability closer to ``1``. To big ``temp`` value
        will make all your probabilities closer to equal values.
    {layer_params}
    """
    function_coef = DictProperty(default={'temp': 1})
    activation_function = softmax


class EuclideDistanceLayer(BaseLayer):
    """ Layer output equal to Euclide distance between input value
    and weights.

    Parameters
    ----------
    {layer_params}
    """
    def output(self, input_value):
        input_data = self.preformat_input(input_value)
        distance = norm(input_data.T - self.weight, axis=0)
        return -reshape(distance, (1, self.weight.shape[1]))


class AngleDistanceLayer(BaseLayer):
    """ Layer which output equal to cosine distance between input value
    and weights.

    Parameters
    ----------
    {layer_params}
    """
    def output(self, input_value):
        input_data = self.preformat_input(input_value)
        norm_prod = norm(input_data) * norm(self.weight, axis=0)
        summated_data = dot(input_data, self.weight)
        return -reshape(arccos(summated_data / norm_prod),
                        (1, self.weight.shape[1]))
