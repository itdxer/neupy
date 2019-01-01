import numpy as np
import tensorflow as tf

from neupy import init
from neupy.utils import asfloat, as_tuple, tf_utils
from neupy.core.properties import (
    NumberProperty, TypedListProperty,
    ParameterProperty, IntProperty,
)
from .base import BaseLayer


__all__ = ('Linear', 'Linear', 'Sigmoid', 'HardSigmoid', 'Tanh',
           'Relu', 'Softplus', 'Softmax', 'Elu', 'PRelu', 'LeakyRelu')


class Linear(BaseLayer):
    """
    The layer with the linear activation function.

    Parameters
    ----------
    size : int or None
        Layer input size. ``None`` means that layer will not create
        parameters and will return only activation function
        output for the specified input value.

    weight : array-like, Tensorfow variable, scalar or Initializer
        Defines layer's weights. Default initialization methods
        you can find :ref:`here <init-methods>`.
        Defaults to :class:`HeNormal() <neupy.init.HeNormal>`.

    bias : 1D array-like, Tensorfow variable, scalar, Initializer or None
        Defines layer's bias. Default initialization methods you can find
        :ref:`here <init-methods>`. Defaults to
        :class:`Constant(0) <neupy.init.Constant>`.
        The ``None`` value excludes bias from the calculations and
        do not add it into parameters list.

    {BaseLayer.name}

    Methods
    -------
    {BaseLayer.Methods}

    activation_function(input)
        Applies activation function to the input.

    Attributes
    ----------
    {BaseLayer.Attributes}

    Examples
    --------
    Linear Regression

    >>> from neupy.layers import *
    >>> network = Input(10) >> Linear(5)
    """
    size = IntProperty(minval=1, allow_none=True)
    weight = ParameterProperty()
    bias = ParameterProperty(allow_none=True)

    def __init__(self, size=None, weight=init.HeNormal(),
                 bias=init.Constant(value=0), name=None):

        self.size = size
        self.weight = weight
        self.bias = bias

        super(Linear, self).__init__(name=name)

    def get_output_shape(self, input_shape):
        if self.size is not None:
            return as_tuple(self.size)
        return input_shape

    def output(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        n_input_features = inputs.shape[-1]

        if self.size is None:
            return self.activation_function(inputs)

        self.weight = self.variable(
            value=self.weight, name='weight',
            shape=as_tuple(n_input_features, self.size),
        )
        output = tf.matmul(inputs, self.weight)

        if self.bias is not None:
            self.bias = self.variable(
                value=self.bias, name='bias',
                shape=as_tuple(self.size),
            )
            output += self.bias

        return self.activation_function(output)

    def activation_function(self, input_value):
        return input_value

    def __repr__(self):
        classname = self.__class__.__name__
        return '{name}({size})'.format(name=classname, size=self.size or '')


class Sigmoid(Linear):
    """
    The layer with the sigmoid activation function.

    Parameters
    ----------
    {Linear.Parameters}

    Methods
    -------
    {Linear.Methods}

    Attributes
    ----------
    {Linear.Attributes}

    Examples
    --------
    Feedforward Neural Networks (FNN)

    >>> from neupy.layers import *
    >>> network = Input(10) >> Sigmoid(5) >> Sigmoid(1)
    """
    def activation_function(self, input_value):
        return tf.nn.sigmoid(input_value)


class HardSigmoid(Linear):
    """
    The layer with the hard sigmoid activation function.

    Parameters
    ----------
    {Linear.Parameters}

    Methods
    -------
    {Linear.Methods}

    Attributes
    ----------
    {Linear.Attributes}

    Examples
    --------
    Feedforward Neural Networks (FNN)

    >>> from neupy.layers import *
    >>> network = Input(10) >> HardSigmoid(5)
    """
    def activation_function(self, input_value):
        input_value = (0.2 * input_value) + 0.5
        return tf.clip_by_value(input_value, 0., 1.)


class Tanh(Linear):
    """
    The layer with the `tanh` activation function.

    Parameters
    ----------
    {Linear.Parameters}

    Methods
    -------
    {Linear.Methods}

    Attributes
    ----------
    {Linear.Attributes}

    Examples
    --------
    Feedforward Neural Networks (FNN)

    >>> from neupy.layers import *
    >>> network = Input(10) >> Tanh(5)
    """
    def activation_function(self, input_value):
        return tf.nn.tanh(input_value)


class Relu(Linear):
    """
    The layer with the rectifier (ReLu) activation function.

    Parameters
    ----------
    {Linear.size}

    alpha : float
        Alpha parameter defines the decreasing rate
        for the negative values. If ``alpha``
        is non-zero value then layer behave like a
        leaky ReLu. Defaults to ``0``.

    weight : array-like, Tensorfow variable, scalar or Initializer
        Defines layer's weights. Default initialization methods
        you can find :ref:`here <init-methods>`.
        Defaults to :class:`HeNormal(gain=2) <neupy.init.HeNormal>`.

    {Linear.bias}

    {BaseLayer.name}

    Methods
    -------
    {Linear.Methods}

    Attributes
    ----------
    {Linear.Attributes}

    Examples
    --------
    Feedforward Neural Networks (FNN)

    >>> from neupy.layers import *
    >>> network = Input(10) >> Relu(20) >> Relu(1)

    Convolutional Neural Networks (CNN)

    >>> from neupy.layers import *
    >>> network = join(
    ...     Input((32, 32, 3)),
    ...     Convolution((3, 3, 16)) >> Relu(),
    ...     Convolution((3, 3, 32)) >> Relu(),
    ...     Reshape(),
    ...     Softmax(10),
    ... )
    """
    alpha = NumberProperty(minval=0)

    def __init__(self, size=None, alpha=0, weight=init.HeNormal(gain=2),
                 bias=init.Constant(value=0), name=None):

        self.alpha = alpha
        super(Relu, self).__init__(
            size=size, weight=weight, bias=bias, name=name)

    def activation_function(self, input_value):
        if self.alpha == 0:
            return tf.nn.relu(input_value)
        return tf.nn.leaky_relu(input_value, asfloat(self.alpha))


class LeakyRelu(Linear):
    """
    The layer with the leaky rectifier (Leaky ReLu)
    activation function.

    Parameters
    ----------
    {Linear.Parameters}

    Methods
    -------
    {Linear.Methods}

    Attributes
    ----------
    {Linear.Attributes}

    Notes
    -----
    Do the same as ``Relu(input_size, alpha=0.01)``.

    Examples
    --------
    Feedforward Neural Networks (FNN)

    >>> from neupy.layers import *
    >>> network = Input(10) >> LeakyRelu(20) >> LeakyRelu(1)
    """
    def activation_function(self, input_value):
        return tf.nn.leaky_relu(input_value, alpha=asfloat(0.01))


class Softplus(Linear):
    """
    The layer with the softplus activation function.

    Parameters
    ----------
    {Linear.Parameters}

    Methods
    -------
    {Linear.Methods}

    Attributes
    ----------
    {Linear.Attributes}

    Examples
    --------
    Feedforward Neural Networks (FNN)

    >>> from neupy.layers import *
    >>> network = Input(10) >> Softplus(4)
    """
    def activation_function(self, input_value):
        return tf.nn.softplus(input_value)


class Softmax(Linear):
    """
    The layer with the softmax activation function.

    Parameters
    ----------
    {Linear.Parameters}

    Methods
    -------
    {Linear.Methods}

    Attributes
    ----------
    {Linear.Attributes}

    Examples
    --------
    Feedforward Neural Networks (FNN)

    >>> from neupy.layers import *
    >>> network = Input(10) >> Relu(20) >> Softmax(10)

    Convolutional Neural Networks (CNN) for Semantic Segmentation

    Softmax layer can be used in order to normalize probabilities
    per pixel. In the example below, we have input 32x32 input image
    with raw prediction  per each pixel for 10 different classes.
    Softmax normalizes raw predictions per pixel to the probability
    distribution.

    >>> from neupy.layers import *
    >>> network = Input((32, 32, 10)) >> Softmax()
    """
    def activation_function(self, input_value):
        return tf.nn.softmax(input_value)


class Elu(Linear):
    """
    The layer with the exponensial linear unit (ELU)
    activation function.

    Parameters
    ----------
    {Linear.Parameters}

    Methods
    -------
    {Linear.Methods}

    Attributes
    ----------
    {Linear.Attributes}

    Examples
    --------
    Feedforward Neural Networks (FNN)

    >>> from neupy.layers import *
    >>> network = Input(10) >> Elu(5) >> Elu(1)

    References
    ----------
    .. [1] http://arxiv.org/pdf/1511.07289v3.pdf
    """
    def activation_function(self, input_value):
        return tf.nn.elu(input_value)


class PRelu(Linear):
    """
    The layer with the parametrized ReLu activation
    function.

    Parameters
    ----------
    alpha_axes : int or tuple
        Axes that will not include unique alpha parameter.
        Single integer value defines the same as a tuple with one value.
        Defaults to ``-1``.

    alpha : array-like, Tensorfow variable, scalar or Initializer
        Alpha parameter per each non-shared axis for the ReLu.
        Scalar value means that each element in the tensor will be
        equal to the specified value.
        Default initialization methods you can find
        :ref:`here <init-methods>`.
        Defaults to ``Constant(value=0.25)``.

    {Linear.Parameters}

    Methods
    -------
    {Linear.Methods}

    Attributes
    ----------
    {Linear.Attributes}

    Examples
    --------
    Feedforward Neural Networks (FNN)

    >>> from neupy.layers import *
    >>> network = Input(10) >> PRelu(20) >> PRelu(1)

    Convolutional Neural Networks (CNN)

    >>> from neupy.layers import *
    >>> network = join(
    ...     Input((32, 32, 3)),
    ...     Convolution((3, 3, 16)) >> PRelu(),
    ...     Convolution((3, 3, 32)) >> PRelu(),
    ...     Reshape(),
    ...     Softmax(10),
    ... )

    References
    ----------
    .. [1] https://arxiv.org/pdf/1502.01852v1.pdf
    """
    alpha_axes = TypedListProperty()
    alpha = ParameterProperty()

    def __init__(self, size=None, alpha=init.Constant(value=0.25),
                 alpha_axes=-1, weight=init.HeNormal(gain=2),
                 bias=init.Constant(value=0), name=None):

        self.alpha = alpha
        self.alpha_axes = as_tuple(alpha_axes)

        if 0 in self.alpha_axes:
            raise ValueError("Cannot specify alpha for 0-axis")

        super(PRelu, self).__init__(
            size=size, weight=weight, bias=bias, name=name)

    def activation_function(self, input_value):
        input_value = tf.convert_to_tensor(input_value, dtype=tf.float32)
        input_shape = input_value.shape
        ndim = len(input_shape)

        if max(self.alpha_axes) >= ndim:
            max_axis_index = len(input_shape) - 1

            raise ValueError(
                "Cannot specify alpha for the axis #{}. Maximum "
                "available axis is {} (0-based indeces)."
                "".format(max(self.alpha_axes), max_axis_index))

        self.alpha = self.variable(
            value=self.alpha,
            name='alpha',
            shape=[input_shape[axis] for axis in self.alpha_axes],
            trainable=True,
        )

        dimensions = np.arange(ndim)
        alpha_axes = dimensions[list(self.alpha_axes)]

        alpha = tf_utils.dimshuffle(self.alpha, ndim, alpha_axes)
        return tf.nn.leaky_relu(tf.to_float(input_value), tf.to_float(alpha))
