import theano
import theano.tensor as T
from numpy import arccos, dot, reshape
from numpy.linalg import norm

from neupy.utils import asfloat
from neupy.core.properties import (NumberBoundProperty, ArrayProperty,
                                   ChoiceProperty)
from neupy.network.utils import step
from neupy.layers.base import BaseLayer
from neupy.layers.utils import GAUSSIAN, VALID_INIT_METHODS, generate_weight


__all__ = ('Layer', 'Linear', 'Sigmoid', 'Step', 'Tanh', 'Relu', 'Softplus',
           'Softmax', 'EuclideDistanceLayer', 'AngleDistanceLayer')


theano_shared_class = T.sharedvar.TensorSharedVariable


class SharedArrayProperty(ArrayProperty):
    expected_type = (ArrayProperty.expected_type, theano_shared_class)


class Layer(BaseLayer):
    """ Base class for input and hidden layers.
    Parameters
    ----------
    {input_size_param}
    {layer_params}
    """

    __layer_params = """weight : 2D array-like or None
        Define your layer weights. ``None`` means that your weights will be
        generate randomly dependence on property ``init_method``.
        ``None`` by default.
    init_method : {'gauss', 'bounded', 'ortho'}
        Weight initialization method.
        ``gauss`` will generate random weights from Standard Normal
        Distribution.
        ``bounded`` generate random weights from Uniform distribution.
        ``ortho`` generate random orthogonal matrix.
        Defaults to ``gauss``.
    bounds : tuple of two float
        Available only for ``init_method`` eqaul to ``bounded``.  Value
        identify minimum and maximum possible value in random weights.
        Defaults to ``(0, 1)``.
    """
    shared_docs = {'layer_params': __layer_params}

    weight = SharedArrayProperty(default=None)
    bias = SharedArrayProperty(default=None)
    bounds = NumberBoundProperty(default=(0, 1))
    init_method = ChoiceProperty(default=GAUSSIAN, choices=VALID_INIT_METHODS)

    def initialize(self):
        output_size = self.relate_to_layer.input_size
        weight = self.weight
        bias = self.bias
        self.step = None

        if not isinstance(weight, theano_shared_class):
            if weight is None:
                weight_shape = (self.input_size, output_size)
                weight = generate_weight(weight_shape, self.bounds,
                                         self.init_method)

            self.weight = theano.shared(value=asfloat(weight), name='weight',
                                        borrow=True)

        if not isinstance(bias, theano_shared_class):
            if bias is None:
                bias_shape = (output_size,)
                bias = generate_weight(bias_shape, self.bounds,
                                       self.init_method)

            self.bias = theano.shared(value=asfloat(bias), name='bias',
                                      borrow=True)

    def output(self, input_value):
        summated = T.dot(input_value, self.weight) + self.bias
        return self.activation_function(summated)


class Linear(Layer):
    """ The layer with the linear activation function.

    Parameters
    ----------
    {input_size_param}
    {layer_params}
    """
    activation_function = (lambda x: x)


class Sigmoid(Layer):
    """ The layer with the sigmoid activation function.

    Parameters
    ----------
    {input_size_param}
    {layer_params}
    """
    activation_function = T.nnet.sigmoid


class Step(Layer):
    """ The layer with the the step activation function.

    Parameters
    ----------
    {input_size_param}
    {layer_params}
    """
    activation_function = step


class Tanh(Layer):
    """ The layer with the `tanh` activation function.

    Parameters
    ----------
    {input_size_param}
    {layer_params}
    """
    activation_function = T.tanh


class Relu(Layer):
    """ The layer with the rectifier activation function.

    Parameters
    ----------
    {input_size_param}
    {layer_params}
    """
    activation_function = T.nnet.relu


class Softplus(Layer):
    """ The layer with the softplus activation function.

    Parameters
    ----------
    {input_size_param}
    {layer_params}
    """
    activation_function = T.nnet.softplus


class Softmax(Layer):
    """ The layer with the softmax activation function.

    Parameters
    ----------
    {input_size_param}
    {layer_params}
    """
    activation_function = T.nnet.softmax


class EuclideDistanceLayer(Layer):
    """ The layer compute Euclide distance between the input
    value and weights.

    Parameters
    ----------
    {input_size_param}
    {layer_params}
    """

    def output(self, input_value):
        distance = norm(input_value.T - self.weight, axis=0)
        return -reshape(distance, (1, self.weight.shape[1]))


class AngleDistanceLayer(Layer):
    """ The layer compute cosine distance between the input value
    and weights.

    Parameters
    ----------
    {input_size_param}
    {layer_params}
    """

    def output(self, input_value):
        norm_prod = norm(input_value) * norm(self.weight, axis=0)
        summated_data = dot(input_value, self.weight)
        return -reshape(arccos(summated_data / norm_prod),
                        (1, self.weight.shape[1]))
