import numpy as np
import tensorflow as tf

from neupy import init
from neupy.utils import asfloat, as_tuple
from neupy.core.properties import (NumberProperty, TypedListProperty,
                                   ParameterProperty, IntProperty)
from .utils import dimshuffle
from .base import ParameterBasedLayer


__all__ = ('ActivationLayer', 'Linear', 'Sigmoid', 'HardSigmoid', 'Tanh',
           'Relu', 'Softplus', 'Softmax', 'Elu', 'PRelu', 'LeakyRelu')


class ActivationLayer(ParameterBasedLayer):
    """
    Base class for the layers based on the activation
    functions.

    Parameters
    ----------
    size : int or None
        Layer input size. ``None`` means that layer will not create
        parameters and will return only activation function
        output for the specified input value.

    {ParameterBasedLayer.weight}

    {ParameterBasedLayer.bias}

    {BaseLayer.Parameters}

    Methods
    -------
    {ParameterBasedLayer.Methods}

    Attributes
    ----------
    {ParameterBasedLayer.Attributes}
    """
    size = IntProperty(minval=1, default=None, allow_none=True)

    def __init__(self, size=None, **options):
        super(ActivationLayer, self).__init__(size=size, **options)

    @property
    def output_shape(self):
        if self.size is not None:
            return as_tuple(self.size)
        return self.input_shape

    def initialize(self):
        if self.size is not None:
            super(ActivationLayer, self).initialize()

    def output(self, input_value):
        if self.size is not None:
            input_value = tf.matmul(input_value, self.weight)

            if self.bias is not None:
                input_value += self.bias

        return self.activation_function(input_value)

    def __repr__(self):
        if self.size is None:
            return super(ParameterBasedLayer, self).__repr__()
        return super(ActivationLayer, self).__repr__()


class Linear(ActivationLayer):
    """
    The layer with the linear activation function.

    Parameters
    ----------
    {ActivationLayer.Parameters}

    Methods
    -------
    {ActivationLayer.Methods}

    Attributes
    ----------
    {ActivationLayer.Attributes}
    """
    def activation_function(self, input_value):
        return input_value


class Sigmoid(ActivationLayer):
    """
    The layer with the sigmoid activation function.

    Parameters
    ----------
    {ActivationLayer.Parameters}

    Methods
    -------
    {ActivationLayer.Methods}

    Attributes
    ----------
    {ActivationLayer.Attributes}
    """
    def activation_function(self, input_value):
        return tf.nn.sigmoid(input_value)


class HardSigmoid(ActivationLayer):
    """
    The layer with the hard sigmoid activation function.

    Parameters
    ----------
    {ActivationLayer.Parameters}

    Methods
    -------
    {ActivationLayer.Methods}

    Attributes
    ----------
    {ActivationLayer.Attributes}
    """
    def activation_function(self, input_value):
        input_value = (0.2 * input_value) + 0.5
        return tf.clip_by_value(input_value, 0., 1.)


class Tanh(ActivationLayer):
    """
    The layer with the `tanh` activation function.

    Parameters
    ----------
    {ActivationLayer.Parameters}

    Methods
    -------
    {ActivationLayer.Methods}

    Attributes
    ----------
    {ActivationLayer.Attributes}
    """
    def activation_function(self, input_value):
        return tf.nn.tanh(input_value)


class Relu(ActivationLayer):
    """
    The layer with the rectifier (ReLu) activation function.

    Parameters
    ----------
    alpha : float
        Alpha parameter defines the decreasing rate
        for the negative values. If ``alpha``
        is non-zero value then layer behave like a
        leaky ReLu. Defaults to ``0``.

    {ActivationLayer.size}

    weight : array-like, Tensorfow variable, scalar or Initializer
        Defines layer's weights. Default initialization methods
        you can find :ref:`here <init-methods>`.
        Defaults to :class:`HeNormal(gain=2) <neupy.init.HeNormal>`.

    {ParameterBasedLayer.bias}

    {BaseLayer.Parameters}

    Methods
    -------
    {ActivationLayer.Methods}

    Attributes
    ----------
    {ActivationLayer.Attributes}
    """
    alpha = NumberProperty(default=0, minval=0)
    weight = ParameterProperty(default=init.HeNormal(gain=2))

    def activation_function(self, input_value):
        if self.alpha == 0:
            return tf.nn.relu(input_value)
        return tf.nn.leaky_relu(input_value, asfloat(self.alpha))


class LeakyRelu(ActivationLayer):
    """
    The layer with the leaky rectifier (Leaky ReLu)
    activation function.

    Parameters
    ----------
    {ActivationLayer.Parameters}

    Methods
    -------
    {ActivationLayer.Methods}

    Attributes
    ----------
    {ActivationLayer.Attributes}

    Notes
    -----
    Do the same as ``layers.Relu(input_size, alpha=0.01)``.
    """
    def activation_function(self, input_value):
        return tf.nn.leaky_relu(input_value, alpha=asfloat(0.01))


class Softplus(ActivationLayer):
    """
    The layer with the softplus activation function.

    Parameters
    ----------
    {ActivationLayer.Parameters}

    Methods
    -------
    {ActivationLayer.Methods}

    Attributes
    ----------
    {ActivationLayer.Attributes}
    """
    def activation_function(self, input_value):
        return tf.nn.softplus(input_value)


class Softmax(ActivationLayer):
    """
    The layer with the softmax activation function.

    Parameters
    ----------
    {ActivationLayer.Parameters}

    Methods
    -------
    {ActivationLayer.Methods}

    Attributes
    ----------
    {ActivationLayer.Attributes}
    """
    def activation_function(self, input_value):
        return tf.nn.softmax(input_value)


class Elu(ActivationLayer):
    """
    The layer with the exponensial linear unit (ELU)
    activation function.

    Parameters
    ----------
    {ActivationLayer.Parameters}

    Methods
    -------
    {ActivationLayer.Methods}

    Attributes
    ----------
    {ActivationLayer.Attributes}

    References
    ----------
    .. [1] http://arxiv.org/pdf/1511.07289v3.pdf
    """
    def activation_function(self, input_value):
        return tf.nn.elu(input_value)


class AxesProperty(TypedListProperty):
    """
    Property defines axes parameter.

    Parameters
    ----------
    {TypedListProperty.n_elements}

    {TypedListProperty.element_type}

    {BaseProperty.default}

    {BaseProperty.required}
    """
    def __set__(self, instance, value):
        value = (value,) if isinstance(value, int) else value
        super(AxesProperty, self).__set__(instance, value)


class PRelu(ActivationLayer):
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

    {ActivationLayer.Parameters}

    Methods
    -------
    {ActivationLayer.Methods}

    Attributes
    ----------
    {ActivationLayer.Attributes}

    References
    ----------
    .. [1] https://arxiv.org/pdf/1502.01852v1.pdf
    """
    alpha_axes = AxesProperty(default=-1)
    alpha = ParameterProperty(default=init.Constant(value=0.25))

    def __init__(self, *args, **options):
        super(PRelu, self).__init__(*args, **options)

        if 0 in self.alpha_axes:
            raise ValueError("Cannot specify alpha for 0-axis")

    def validate(self, input_shape):
        if max(self.alpha_axes) > len(input_shape):
            max_axis_index = len(input_shape) - 1
            raise ValueError("Cannot specify alpha for the axis #{}. "
                             "Maximum available axis is {} (0-based indeces)."
                             "".format(max(self.alpha_axes), max_axis_index))

    def initialize(self):
        super(PRelu, self).initialize()
        output_shape = as_tuple(None, self.output_shape)

        alpha_shape = [output_shape[axis] for axis in self.alpha_axes]
        self.add_parameter(
            value=self.alpha,
            name='alpha',
            shape=alpha_shape,
            trainable=True,
        )

    def activation_function(self, input_value):
        input_value = tf.convert_to_tensor(input_value, dtype=tf.float32)
        ndim = len(input_value.get_shape())

        dimensions = np.arange(ndim)
        alpha_axes = dimensions[list(self.alpha_axes)]

        alpha = dimshuffle(self.alpha, ndim, alpha_axes)
        return tf.nn.leaky_relu(tf.to_float(input_value), tf.to_float(alpha))
