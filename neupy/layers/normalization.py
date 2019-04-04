import tensorflow as tf

from neupy.core.properties import (
    ProperFractionProperty,
    ParameterProperty,
    TypedListProperty,
    NumberProperty,
    IntProperty,
)
from neupy.utils import asfloat
from neupy.exceptions import (
    WeightInitializationError,
    LayerConnectionError,
)
from .base import Identity


__all__ = ('BatchNorm', 'LocalResponseNorm', 'GroupNorm')


class BatchNorm(Identity):
    """
    Batch normalization layer.

    Parameters
    ----------
    axes : tuple with ints or None
        Axes along which normalization will be applied. The ``None``
        value means that normalization will be applied over all axes
        except the last one. In case of 4D tensor it will
        be equal to ``(0, 1, 2)``. Defaults to ``None``.

    epsilon : float
        Epsilon is a positive constant that adds to the standard
        deviation to prevent the division by zero.
        Defaults to ``1e-5``.

    alpha : float
        Coefficient for the exponential moving average of
        batch-wise means and standard deviations computed during
        training; the closer to one, the more it will depend on
        the last batches seen. Value needs to be between ``0`` and ``1``.
        Defaults to ``0.1``.

    gamma : array-like, Tensorfow variable, scalar or Initializer
        Scale. Default initialization methods you can
        find :ref:`here <init-methods>`.
        Defaults to ``Constant(value=1)``.

    beta : array-like, Tensorfow variable, scalar or Initializer
        Offset. Default initialization methods you can
        find :ref:`here <init-methods>`.
        Defaults to ``Constant(value=0)``.

    running_mean : array-like, Tensorfow variable, scalar or Initializer
        Default initialization methods you can
        find :ref:`here <init-methods>`.
        Defaults to ``Constant(value=0)``.

    running_inv_std : array-like, Tensorfow variable, scalar or Initializer
        Default initialization methods you can
        find :ref:`here <init-methods>`.
        Defaults to ``Constant(value=1)``.

    {Identity.name}

    Methods
    -------
    {Identity.Methods}

    Attributes
    ----------
    {Identity.Attributes}

    Examples
    --------

    Feedforward Neural Networks (FNN) with batch normalization after
    activation function was applied.

    >>> from neupy.layers import *
    >>> network = join(
    ...     Input(10),
    ...     Relu(5) >> BatchNorm(),
    ...     Relu(5) >> BatchNorm(),
    ...     Sigmoid(1),
    ... )

    Feedforward Neural Networks (FNN) with batch normalization before
    activation function was applied.

    >>> from neupy.layers import *
    >>> network = join(
    ...     Input(10),
    ...     Linear(5) >> BatchNorm() >> Relu(),
    ...     Linear(5) >> BatchNorm() >> Relu(),
    ...     Sigmoid(1),
    ... )

    Convolutional Neural Networks (CNN)

    >>> from neupy.layers import *
    >>> network = join(
    ...     Input((28, 28, 1)),
    ...     Convolution((3, 3, 16)) >> BatchNorm() >> Relu(),
    ...     Convolution((3, 3, 16)) >> BatchNorm() >> Relu(),
    ...     Reshape(),
    ...     Softmax(10),
    ... )

    References
    ----------
    .. [1] Batch Normalization: Accelerating Deep Network Training
           by Reducing Internal Covariate Shift,
           http://arxiv.org/pdf/1502.03167v3.pdf
    """
    axes = TypedListProperty(allow_none=True)
    epsilon = NumberProperty(minval=0)
    alpha = ProperFractionProperty()
    beta = ParameterProperty()
    gamma = ParameterProperty()

    running_mean = ParameterProperty()
    running_inv_std = ParameterProperty()

    def __init__(self, axes=None, alpha=0.1, beta=0, gamma=1, epsilon=1e-5,
                 running_mean=0, running_inv_std=1, name=None):

        super(BatchNorm, self).__init__(name=name)

        self.axes = axes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.running_mean = running_mean
        self.running_inv_std = running_inv_std

        if axes is not None and len(set(axes)) != len(axes):
            raise ValueError(
                "Specified axes have to contain only unique values")

    def create_variables(self, input_shape):
        input_shape = tf.TensorShape(input_shape)

        if input_shape.ndims is None:
            raise WeightInitializationError(
                "Cannot initialize variables for the batch normalization "
                "layer, because input shape is undefined. Layer: {}"
                "".format(self))

        if self.axes is None:
            # If ndims == 4 then axes = (0, 1, 2)
            # If ndims == 2 then axes = (0,)
            self.axes = tuple(range(input_shape.ndims - 1))

        if any(axis >= input_shape.ndims for axis in self.axes):
            raise LayerConnectionError(
                "Batch normalization cannot be applied over one of "
                "the axis, because input has only {} dimensions. Layer: {}"
                "".format(input_shape.ndims, self))

        parameter_shape = tuple([
            input_shape[axis].value if axis not in self.axes else 1
            for axis in range(input_shape.ndims)
        ])

        if any(parameter is None for parameter in parameter_shape):
            unknown_dim_index = parameter_shape.index(None)

            raise WeightInitializationError(
                "Cannot create variables for batch normalization, because "
                "input has unknown dimension #{} (0-based indices). "
                "Input shape: {}, Layer: {}".format(
                    unknown_dim_index, input_shape, self))

        self.input_shape = input_shape
        self.running_mean = self.variable(
            value=self.running_mean, shape=parameter_shape,
            name='running_mean', trainable=False)

        self.running_inv_std = self.variable(
            value=self.running_inv_std, shape=parameter_shape,
            name='running_inv_std', trainable=False)

        self.gamma = self.variable(
            value=self.gamma, name='gamma',
            shape=parameter_shape)

        self.beta = self.variable(
            value=self.beta, name='beta',
            shape=parameter_shape)

    def output(self, input, training=False):
        input = tf.convert_to_tensor(input, dtype=tf.float32)

        if not training:
            mean = self.running_mean
            inv_std = self.running_inv_std
        else:
            alpha = asfloat(self.alpha)
            mean = tf.reduce_mean(
                input, self.axes,
                keepdims=True, name="mean",
            )
            variance = tf.reduce_mean(
                tf.squared_difference(input, tf.stop_gradient(mean)),
                self.axes,
                keepdims=True,
                name="variance",
            )
            inv_std = tf.rsqrt(variance + asfloat(self.epsilon))

            tf.add_to_collection(
                tf.GraphKeys.UPDATE_OPS,
                self.running_inv_std.assign(
                    asfloat(1 - alpha) * self.running_inv_std + alpha * inv_std
                )
            )
            tf.add_to_collection(
                tf.GraphKeys.UPDATE_OPS,
                self.running_mean.assign(
                    asfloat(1 - alpha) * self.running_mean + alpha * mean
                )
            )

        normalized_value = (input - mean) * inv_std
        return self.gamma * normalized_value + self.beta


class LocalResponseNorm(Identity):
    """
    Local Response Normalization Layer.

    Aggregation is purely across channels, not within channels,
    and performed "pixelwise".

    If the value of the :math:`i` th channel is :math:`x_i`, the output is

    .. math::
        x_i = \\frac{{x_i}}{{ (k + ( \\alpha \\sum_j x_j^2 ))^\\beta }}

    where the summation is performed over this position on :math:`n`
    neighboring channels.

    Parameters
    ----------
    alpha : float
        Coefficient, see equation above. Defaults to ``1e-4``.

    beta : float
        Offset, see equation above. Defaults to ``0.75``.

    k : float
        Exponent, see equation above. Defaults to ``2``.

    depth_radius : int
        Number of adjacent channels to normalize over, must be odd.
        Defaults to ``5``.

    {Identity.name}

    Methods
    -------
    {Identity.Methods}

    Attributes
    ----------
    {Identity.Attributes}

    Examples
    --------
    >>> from neupy.layers import *
    >>> network = Input((10, 10, 12)) >> LocalResponseNorm()
    """
    alpha = NumberProperty()
    beta = NumberProperty()
    k = NumberProperty()
    depth_radius = IntProperty()

    def __init__(self, alpha=1e-4, beta=0.75, k=2, depth_radius=5, name=None):
        super(LocalResponseNorm, self).__init__(name=name)

        if depth_radius % 2 == 0:
            raise ValueError("Only works with odd `depth_radius` values")

        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.depth_radius = depth_radius

    def get_output_shape(self, input_shape):
        if input_shape and input_shape.ndims != 4:
            raise LayerConnectionError(
                "Layer `{}` expected input with 4 dimensions, got {} instead. "
                "Shape: {}".format(self.name, input_shape.ndims, input_shape))

        return super(LocalResponseNorm, self).get_output_shape(input_shape)

    def output(self, input, **kwargs):
        return tf.nn.local_response_normalization(
            input,
            depth_radius=self.depth_radius,
            bias=self.k,
            alpha=self.alpha,
            beta=self.beta)


class GroupNorm(Identity):
    """
    Group Normalization layer. This layer is a simple alternative to the
    Batch Normalization layer for cases when batch size is small.

    Parameters
    ----------
    n_groups : int
        During normalization all the channels will be break down into
        separate groups and mean and variance will be estimated per group.
        This parameter controls number of groups.

    gamma : array-like, Tensorfow variable, scalar or Initializer
        Scale. Default initialization methods you can
        find :ref:`here <init-methods>`.
        Defaults to ``Constant(value=1)``.

    beta : array-like, Tensorfow variable, scalar or Initializer
        Offset. Default initialization methods you can
        find :ref:`here <init-methods>`.
        Defaults to ``Constant(value=0)``.

    epsilon : float
        Epsilon ensures that input rescaling procedure that uses estimated
        variance will never cause division by zero. Defaults to ``1e-5``.

    {Identity.name}

    Methods
    -------
    {Identity.Methods}

    Attributes
    ----------
    {Identity.Attributes}

    Examples
    --------
    Convolutional Neural Networks (CNN)

    >>> from neupy.layers import *
    >>> network = join(
    ...     Input((28, 28, 1)),
    ...     Convolution((3, 3, 16)) >> GroupNorm(4) >> Relu(),
    ...     Convolution((3, 3, 16)) >> GroupNorm(4) >> Relu(),
    ...     Reshape(),
    ...     Softmax(10),
    ... )

    References
    ----------
    .. [1] Group Normalization, Yuxin Wu, Kaiming He,
           https://arxiv.org/pdf/1803.08494.pdf
    """
    n_groups = IntProperty(minval=1)
    beta = ParameterProperty()
    gamma = ParameterProperty()
    epsilon = NumberProperty(minval=0)

    def __init__(self, n_groups, beta=0, gamma=1, epsilon=1e-5, name=None):
        super(GroupNorm, self).__init__(name=name)

        self.n_groups = n_groups
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def create_variables(self, input_shape):
        n_channels = input_shape[3]

        if n_channels.value is None:
            raise WeightInitializationError(
                "Cannot initialize variables when number of "
                "channels is unknown. Input shape: {}, Layer: {}"
                "".format(input_shape, self))

        parameter_shape = (1, 1, 1, n_channels)

        self.gamma = self.variable(
            value=self.gamma, name='gamma',
            shape=parameter_shape)

        self.beta = self.variable(
            value=self.beta, name='beta',
            shape=parameter_shape)

    def get_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)

        if input_shape and input_shape.ndims != 4:
            raise LayerConnectionError(
                "Group normalization layer expects 4 dimensional input, "
                "got {} instead. Input shape: {}, Layer: {}"
                "".format(input_shape.ndims, input_shape, self))

        n_channels = input_shape[3]

        if n_channels.value and n_channels % self.n_groups != 0:
            raise LayerConnectionError(
                "Cannot divide {} input channels into {} groups. "
                "Input shape: {}, Layer: {}".format(
                    n_channels, self.n_groups, input_shape, self))

        return super(GroupNorm, self).get_output_shape(input_shape)

    def output(self, input):
        input = tf.convert_to_tensor(input, dtype=tf.float32)
        input_shape = tf.shape(input)
        n_groups = self.n_groups

        # We access dimensional information in form of tensors in case
        # if some of the dimensions are undefined. In this way we make
        # sure that reshape will work even if part of the input shape
        # is undefined.
        dims = [input_shape[i] for i in range(4)]
        n_samples, height, width, n_channels = dims

        input = tf.reshape(input, [
            n_samples, height, width, n_groups, n_channels // n_groups])

        mean, variance = tf.nn.moments(input, [1, 2, 4], keep_dims=True)
        input = (input - mean) / tf.sqrt(variance + self.epsilon)
        input = tf.reshape(input, input_shape)

        return input * self.gamma + self.beta
