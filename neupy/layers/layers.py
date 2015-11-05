from functools import partial

import theano.tensor as T
from numpy import arccos, dot, reshape
from numpy.linalg import norm

from neupy.network.utils import step
from neupy.core.properties import DictProperty
from neupy.layers.base import BaseLayer


__all__ = ('Layer', 'Linear', 'Sigmoid', 'Step', 'Tanh', 'Relu', 'Softplus',
           'Softmax', 'EuclideDistanceLayer', 'AngleDistanceLayer')


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
            partial_func = partial(self.activation_function)
            self.activation_function = partial_func(
                self.activation_function, **self.function_coef
            )


class Linear(Layer):
    """ The layer with the linear activation function.

    Parameters
    ----------
    {layer_params}
    """
    activation_function = (lambda x: x)


class Sigmoid(Layer):
    """ The layer with the sigmoid activation function.

    Parameters
    ----------
    function_coef : dict
        The default configurations for the sigmoid activation function.
        There is one available parameter ``alpha`` (defaults to ``1``).
        Parameter ``alpha`` controls function shape.
    {layer_params}
    """
    # function_coef = DictProperty(default={'alpha': 1})
    activation_function = T.nnet.sigmoid


class Step(Layer):
    """ The layer with the the step activation function.

    Parameters
    ----------
    {layer_params}
    """
    activation_function = step


class Tanh(Layer):
    """ The layer with the `tanh` activation function.

    Parameters
    ----------
    function_coef : dict
        The default configurations for the sigmoid activation function.
        There is one available parameter ``alpha`` (defaults to ``1``).
        Parameter `alpha` controls function shape.
    {layer_params}
    """
    # function_coef = DictProperty(default={'alpha': 1})
    activation_function = T.tanh


class Relu(Layer):
    """ The layer with the rectifier activation function.

    Parameters
    ----------
    {layer_params}
    """
    activation_function = T.nnet.relu


class Softplus(Layer):
    """ The layer with the softplus activation function.

    Parameters
    ----------
    {layer_params}
    """
    activation_function = T.nnet.softplus


class Softmax(Layer):
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
    # function_coef = DictProperty(default={'temp': 1})
    activation_function = T.nnet.softmax


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
