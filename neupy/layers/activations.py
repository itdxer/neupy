import numpy as np
import tensorflow as tf

from neupy import init
from neupy.utils import asfloat, as_tuple, tf_utils
from neupy.exceptions import LayerConnectionError, WeightInitializationError
from neupy.core.properties import (
    NumberProperty, TypedListProperty,
    ParameterProperty, IntProperty,
)
from .base import BaseLayer


__all__ = (
    'Linear', 'Sigmoid', 'Tanh', 'Softmax',
    'Relu', 'LeakyRelu', 'Elu', 'PRelu',
    'Softplus', 'HardSigmoid',
)


class Linear(BaseLayer):
    """
    Layer with linear activation function. It applies linear transformation
    when the ``n_units`` parameter specified and acts as an identity
    when it's not specified.

    Parameters
    ----------
    n_units : int or None
        Number of units in the layers. It also corresponds to the number of
        output features that will be produced per sample after passing it
        through this layer. The ``None`` value means that layer will not have
        parameters and it will only apply activation function to the input
        without linear transformation output for the specified input value.
        Defaults to ``None``.

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
    n_units = IntProperty(minval=1, allow_none=True)
    weight = ParameterProperty()
    bias = ParameterProperty(allow_none=True)

    def __init__(self, n_units=None, weight=init.HeNormal(), bias=0,
                 name=None):

        super(Linear, self).__init__(name=name)

        self.n_units = n_units
        self.weight = weight
        self.bias = bias

    def get_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)

        if self.n_units is None:
            return input_shape

        if input_shape and input_shape.ndims != 2:
            raise LayerConnectionError(
                "Input shape expected to have 2 dimensions, got {} instead. "
                "Shape: {}".format(input_shape.ndims, input_shape))

        n_samples = input_shape[0]
        return tf.TensorShape((n_samples, self.n_units))

    def create_variables(self, input_shape):
        if self.n_units is None:
            return

        input_shape = tf.TensorShape(input_shape)
        self.input_shape = input_shape
        _, n_input_features = input_shape

        if n_input_features.value is None:
            raise WeightInitializationError(
                "Cannot create variables for the layer `{}`, because "
                "number of input features is unknown. Input shape: {}"
                "Layer: {}".format(self.name, input_shape, self))

        self.weight = self.variable(
            value=self.weight, name='weight',
            shape=as_tuple(n_input_features, self.n_units))

        if self.bias is not None:
            self.bias = self.variable(
                value=self.bias, name='bias',
                shape=as_tuple(self.n_units))

    def output(self, input, **kwargs):
        input = tf.convert_to_tensor(input, dtype=tf.float32)

        if self.n_units is None:
            return self.activation_function(input)

        if self.bias is None:
            output = tf.matmul(input, self.weight)
            return self.activation_function(output)

        output = tf.matmul(input, self.weight) + self.bias
        return self.activation_function(output)

    def activation_function(self, input_value):
        return input_value

    def __repr__(self):
        if self.n_units is None:
            return self._repr_arguments(name=self.name)

        return self._repr_arguments(
            self.n_units,
            name=self.name,
            weight=self.weight,
            bias=self.bias,
        )


class Sigmoid(Linear):
    """
    Layer with the sigmoid used as an activation function. It applies
    linear transformation when the ``n_units`` parameter specified and
    sigmoid function after the transformation. When ``n_units`` is not
    specified, only sigmoid function will be applied to the input.

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
    Logistic Regression (LR)

    >>> from neupy.layers import *
    >>> network = Input(10) >> Sigmoid(1)

    Feedforward Neural Networks (FNN)

    >>> from neupy.layers import *
    >>> network = Input(10) >> Sigmoid(5) >> Sigmoid(1)

    Convolutional Neural Networks (CNN) for Semantic Segmentation

    Sigmoid layer can be used in order to normalize probabilities
    per pixel in semantic classification task with two classes.
    In the example below, we have as input 32x32 image that predicts
    one of the two classes. Sigmoid normalizes raw predictions per pixel
    to the valid probabilities.

    >>> from neupy.layers import *
    >>> network = Input((32, 32, 1)) >> Sigmoid()
    """
    def activation_function(self, input_value):
        return tf.nn.sigmoid(input_value)


class HardSigmoid(Linear):
    """
    Layer with the hard sigmoid used as an activation function. It applies
    linear transformation when the ``n_units`` parameter specified and
    hard sigmoid function after the transformation. When ``n_units`` is
    not specified, only hard sigmoid function will be applied to the input.

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
    Layer with the hyperbolic tangent used as an activation function.
    It applies linear transformation when the ``n_units`` parameter
    specified and ``tanh`` function after the transformation. When
    ``n_units`` is not specified, only ``tanh`` function will be applied
    to the input.

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
    Layer with the rectifier (ReLu) used as an activation function.
    It applies linear transformation when the ``n_units`` parameter
    specified and ``relu`` function after the transformation. When
    ``n_units`` is not specified, only ``relu`` function will be applied
    to the input.

    Parameters
    ----------
    {Linear.n_units}

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

    def __init__(self, n_units=None, alpha=0, weight=init.HeNormal(gain=2),
                 bias=init.Constant(value=0), name=None):

        self.alpha = alpha
        super(Relu, self).__init__(
            n_units=n_units, weight=weight, bias=bias, name=name)

    def activation_function(self, input_value):
        if self.alpha == 0:
            return tf.nn.relu(input_value)
        return tf.nn.leaky_relu(input_value, asfloat(self.alpha))

    def __repr__(self):
        if self.n_units is None:
            return self._repr_arguments(name=self.name, alpha=self.alpha)

        return self._repr_arguments(
            self.n_units,
            name=self.name,
            alpha=self.alpha,
            weight=self.weight,
            bias=self.bias,
        )


class LeakyRelu(Linear):
    """
    Layer with the leaky rectifier (Leaky ReLu) used as an activation
    function. It applies linear transformation when the ``n_units``
    parameter specified and leaky relu function after the transformation.
    When ``n_units`` is not specified, only leaky relu function will be
    applied to the input.

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
    Layer with the softplus used as an activation function. It applies linear
    transformation when the ``n_units`` parameter specified and softplus
    function after the transformation. When ``n_units`` is not specified,
    only softplus function will be applied to the input.

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
    Layer with the softmax activation function. It applies linear
    transformation when the ``n_units`` parameter specified and softmax
    function after the transformation. When ``n_units`` is not specified,
    only softmax function will be applied to the input.

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
    per pixel. In the example below, we have as input 32x32 image
    with raw prediction per each pixel for 10 different classes.
    Softmax normalizes raw predictions per pixel to the probability
    distribution.

    >>> from neupy.layers import *
    >>> network = Input((32, 32, 10)) >> Softmax()
    """
    def activation_function(self, input_value):
        return tf.nn.softmax(input_value)


class Elu(Linear):
    """
    Layer with the exponential linear unit (ELU) used as an activation
    function. It applies linear transformation when the ``n_units``
    parameter specified and elu function after the transformation.
    When ``n_units`` is not specified, only elu function will be
    applied to the input.

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
    Layer with the parametrized ReLu used as an activation function.
    Layer learns additional parameter ``alpha`` during the training.

    It applies linear transformation when the ``n_units`` parameter
    specified and parametrized relu function after the transformation.
    When ``n_units`` is not specified, only parametrized relu function
    will be applied to the input.

    Parameters
    ----------
    alpha_axes : int or tuple
        Axes that will not include unique alpha parameter.
        Single integer value defines the same as a tuple with one value.
        Defaults to ``-1``.

    alpha : array-like, Tensorfow variable, scalar or Initializer
        Separate alpha parameter per each non-shared axis for the ReLu.
        Scalar value means that each element in the tensor will be
        equal to the specified value. Default initialization methods you
        can find :ref:`here <init-methods>`.
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
    .. [1] Delving Deep into Rectifiers: Surpassing Human-Level
           Performance on ImageNet Classification.
           https://arxiv.org/pdf/1502.01852v1.pdf
    """
    alpha_axes = TypedListProperty()
    alpha = ParameterProperty()

    def __init__(self, n_units=None, alpha_axes=-1, alpha=0.25,
                 weight=init.HeNormal(gain=2), bias=0, name=None):

        self.alpha = alpha
        self.alpha_axes = as_tuple(alpha_axes)

        if 0 in self.alpha_axes:
            raise ValueError("Cannot specify alpha for 0-axis")

        super(PRelu, self).__init__(
            n_units=n_units, weight=weight, bias=bias, name=name)

    def get_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)

        if input_shape and max(self.alpha_axes) >= input_shape.ndims:
            max_axis_index = input_shape.ndims - 1

            raise LayerConnectionError(
                "Cannot specify alpha for the axis #{}. Maximum "
                "available axis is {} (0-based indices)."
                "".format(max(self.alpha_axes), max_axis_index))

        return super(PRelu, self).get_output_shape(input_shape)

    def create_variables(self, input_shape):
        super(PRelu, self).create_variables(input_shape)
        output_shape = self.get_output_shape(input_shape)

        self.alpha = self.variable(
            value=self.alpha, name='alpha',
            shape=[output_shape[axis] for axis in self.alpha_axes])

    def activation_function(self, input):
        input = tf.convert_to_tensor(input, dtype=tf.float32)
        ndim = input.shape.ndims

        dimensions = np.arange(ndim)
        alpha_axes = dimensions[list(self.alpha_axes)]

        alpha = tf_utils.dimshuffle(self.alpha, ndim, alpha_axes)
        return tf.maximum(0.0, input) + alpha * tf.minimum(0.0, input)

    def __repr__(self):
        if self.n_units is None:
            return self._repr_arguments(
                name=self.name,
                alpha_axes=self.alpha_axes,
                alpha=self.alpha)

        return self._repr_arguments(
            self.n_units,
            name=self.name,
            alpha_axes=self.alpha_axes,
            alpha=self.alpha,
            weight=self.weight,
            bias=self.bias)
