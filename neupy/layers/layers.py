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
    """ The base class of neural network layers.

    Parameters
    ----------
    function_coef : dict
        The default settings for the activation function.
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
    """ The layer with the linear activation function.

    Parameters
    ----------
    {layer_params}
    """
    activation_function = linear


class SigmoidLayer(Layer):
    """ The layer with the sigmoid activation function.

    Parameters
    ----------
    function_coef : dict
        The default configurations for the sigmoid activation function.
        There is one available parameter ``alpha`` (defaults to ``1``).
        Parameter ``alpha`` controls function shape.
    {layer_params}
    """
    function_coef = DictProperty(default={'alpha': 1})
    activation_function = sigmoid


class StepLayer(Layer):
    """ The layer with the the step activation function.

    Parameters
    ----------
    {layer_params}
    """
    activation_function = step


class TanhLayer(Layer):
    """ The layer with the `tanh` activation function.

    Parameters
    ----------
    function_coef : dict
        The default configurations for the sigmoid activation function.
        There is one available parameter ``alpha`` (defaults to ``1``).
        Parameter `alpha` controls function shape.
    {layer_params}
    """
    function_coef = DictProperty(default={'alpha': 1})
    activation_function = tanh


class RectifierLayer(Layer):
    """ The layer with the rectifier activation function.

    Parameters
    ----------
    {layer_params}
    """
    activation_function = rectifier


class SoftplusLayer(Layer):
    """ The layer with the softplus activation function.

    Parameters
    ----------
    {layer_params}
    """
    activation_function = softplus


class SoftmaxLayer(Layer):
    """ The layer with the softmax activation function.

    Parameters
    ----------
    function_coef : dict
        The default configurations for the softmax activation function.
        There is one available parameter ``temp`` (defaults to ``1``).
        Lower ``temp`` value will make your winner probability closer
        to ``1``. Higher ``temp`` value will make all probabilities
        values equal to each other.
    {layer_params}
    """
    function_coef = DictProperty(default={'temp': 1})
    activation_function = softmax


class EuclideDistanceLayer(BaseLayer):
    """ The layer compute Euclide distance between the input
    value and weights.

    Parameters
    ----------
    {layer_params}
    """
    def output(self, input_value):
        input_data = self.preformat_input(input_value)
        distance = norm(input_data.T - self.weight, axis=0)
        return -reshape(distance, (1, self.weight.shape[1]))


class AngleDistanceLayer(BaseLayer):
    """ The layer compute cosine distance between the input value
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
