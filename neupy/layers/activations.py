import six
import theano.tensor as T

from neupy.core.config import ConfigMeta
from .base import ParameterBasedLayer


__all__ = ('ActivationLayer', 'Linear', 'Sigmoid', 'HardSigmoid', 'Step',
           'Tanh', 'Relu', 'Softplus', 'Softmax')


class LayerMeta(ConfigMeta):
    """ Meta-class overrides activation functions an make them
    behave as usual functions.
    """
    def __new__(cls, clsname, bases, attrs):
        if 'activation_function' in attrs:
            # Override `activation_function` in `staticmethod` by default.
            # Python 2 translate any assigned function as class method
            # and try call it with with argument `self` which broke
            # logic. For this reason we try make it static.
            attrs['activation_function'] = staticmethod(
                attrs['activation_function']
            )
        return super(LayerMeta, cls).__new__(cls, clsname, bases, attrs)


class ActivationLayer(six.with_metaclass(LayerMeta, ParameterBasedLayer)):
    """ Base class for the layers based on the activation
    functions.

    Parameters
    ----------
    size : int or None
        Layer input size. ``None`` means that layer will not create
        parameters and will return only activation function
        output for the specified input value.
    {ParameterBasedLayer.weight}
    {ParameterBasedLayer.bias}
    {ParameterBasedLayer.init_method}
    {ParameterBasedLayer.bounds}
    """
    def __init__(self, size=None, **options):
        # If you set class method function variable, python will interpret
        # it as a new class method and will call it with a `self`
        # argument.
        if hasattr(self.__class__, 'activation_function'):
            self.activation_function = self.__class__.activation_function
        super(ActivationLayer, self).__init__(size, **options)

    def initialize(self):
        if self.size is not None:
            super(ActivationLayer, self).initialize()

    def output(self, input_value):
        if self.size is not None:
            input_value = T.dot(input_value, self.weight) + self.bias
        return self.activation_function(input_value)

    def __repr__(self):
        if self.size is None:
            return super(ParameterBasedLayer, self).__repr__()
        return super(ActivationLayer, self).__repr__()


class Linear(ActivationLayer):
    """ The layer with the linear activation function.

    Parameters
    ----------
    {ActivationLayer.size}
    {ParameterBasedLayer.weight}
    {ParameterBasedLayer.bias}
    {ParameterBasedLayer.init_method}
    {ParameterBasedLayer.bounds}
    """
    activation_function = (lambda x: x)


class Sigmoid(ActivationLayer):
    """ The layer with the sigmoid activation function.

    Parameters
    ----------
    {ActivationLayer.size}
    {ParameterBasedLayer.weight}
    {ParameterBasedLayer.bias}
    {ParameterBasedLayer.init_method}
    {ParameterBasedLayer.bounds}
    """
    activation_function = T.nnet.sigmoid


class HardSigmoid(ActivationLayer):
    """ The layer with the hard sigmoid activation function.

    Parameters
    ----------
    {ActivationLayer.size}
    {ParameterBasedLayer.weight}
    {ParameterBasedLayer.bias}
    {ParameterBasedLayer.init_method}
    {ParameterBasedLayer.bounds}
    """
    activation_function = T.nnet.hard_sigmoid


def step_function(value):
    """ Step activation function.

    Parameters
    ----------
    value : symbolic tensor
        Tensor to compute the activation function for.
    alpha : scalar or tensor, optional
        Slope for negative input, usually between 0 and 1. The
        default value of 0 will lead to the standard rectifier, 1 will lead to
        a linear activation function, and any value in between will give a
        leaky rectifier. A shared variable (broadcastable against `x`) will
        result in a parameterized rectifier with learnable slope(s).

    Returns
    -------
    symbolic tensor
        Element-wise rectifier applied to `x`.
    """
    return T.gt(value, 0)


class Step(ActivationLayer):
    """ The layer with the the step activation function.

    Parameters
    ----------
    {ActivationLayer.size}
    {ParameterBasedLayer.weight}
    {ParameterBasedLayer.bias}
    {ParameterBasedLayer.init_method}
    {ParameterBasedLayer.bounds}
    """
    activation_function = step_function


class Tanh(ActivationLayer):
    """ The layer with the `tanh` activation function.

    Parameters
    ----------
    {ActivationLayer.size}
    {ParameterBasedLayer.weight}
    {ParameterBasedLayer.bias}
    {ParameterBasedLayer.init_method}
    {ParameterBasedLayer.bounds}
    """
    activation_function = T.tanh


class Relu(ActivationLayer):
    """ The layer with the rectifier activation function.

    Parameters
    ----------
    {ActivationLayer.size}
    {ParameterBasedLayer.weight}
    {ParameterBasedLayer.bias}
    {ParameterBasedLayer.init_method}
    {ParameterBasedLayer.bounds}
    """
    activation_function = T.nnet.relu


class Softplus(ActivationLayer):
    """ The layer with the softplus activation function.

    Parameters
    ----------
    {ActivationLayer.size}
    {ParameterBasedLayer.weight}
    {ParameterBasedLayer.bias}
    {ParameterBasedLayer.init_method}
    {ParameterBasedLayer.bounds}
    """
    activation_function = T.nnet.softplus


class Softmax(ActivationLayer):
    """ The layer with the softmax activation function.

    Parameters
    ----------
    {ActivationLayer.size}
    {ParameterBasedLayer.weight}
    {ParameterBasedLayer.bias}
    {ParameterBasedLayer.init_method}
    {ParameterBasedLayer.bounds}
    """
    activation_function = T.nnet.softmax
