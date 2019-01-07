import tensorflow as tf

from neupy.core.properties import (
    ProperFractionProperty,
    ParameterProperty,
    TypedListProperty,
    NumberProperty,
    IntProperty,
)
from neupy.utils import asfloat, as_tuple
from neupy.exceptions import LayerConnectionError
from .base import Identity


__all__ = ('BatchNorm', 'LocalResponseNorm')


def find_opposite_axes(axes, ndim):
    """
    Based on the total number of dimensions function
    finds all axes that are missed in the specified
    list ``axes``.

    Parameters
    ----------
    axes : list or tuple
        Already known axes.

    ndim : int
        Total number of dimensions.

    Returns
    -------
    list

    Examples
    --------
    >>> from neupy.layers.normalization import find_opposite_axes
    >>> find_opposite_axes([0, 1], ndim=4)
    [2, 3]
    >>>
    >>> find_opposite_axes([], ndim=4)
    [0, 1, 2, 3]
    >>>
    >>> find_opposite_axes([0, 1, 2], ndim=3)
    []
    """
    if any(axis >= ndim for axis in axes):
        raise ValueError(
            "Some axes have invalid values. Axis value "
            "should be between 0 and {}".format(ndim))

    return [axis for axis in range(ndim) if axis not in axes]


class BatchNorm(Identity):
    """
    Batch-normalization layer.

    Parameters
    ----------
    axes : int, tuple with int or None
        The axis or axes along which normalization is applied.
        ``None`` means that normalization will be applied over
        all axes except the first one. In case of 4D tensor it will
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
        Default initialization methods you can
        find :ref:`here <init-methods>`.
        Defaults to ``Constant(value=1)``.

    beta : array-like, Tensorfow variable, scalar or Initializer
        Default initialization methods you can
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

        if axes is not None and 0 in axes:
            raise ValueError(
                "Cannot specify axes for batch dimension (0-axis)")

    def initialize_variables(self, input):
        input_shape = input.shape
        ndim = len(input_shape)

        if self.axes is None:
            # If ndim == 4 then axes = (0, 1, 2)
            # If ndim == 2 then axes = (0,)
            self.axes = tuple(range(ndim - 1))

        if any(axis >= ndim for axis in self.axes):
            raise ValueError(
                "Cannot apply batch normalization "
                "on the axis that doesn't exist.")

        opposite_axes = find_opposite_axes(self.axes, ndim)
        parameter_shape = [
            input_shape[axis] if axis in opposite_axes else 1
            for axis in range(ndim)
        ]

        if any(parameter is None for parameter in parameter_shape):
            unknown_dim_index = parameter_shape.index(None)
            raise ValueError(
                "Cannot apply batch normalization on the axis "
                "with unknown size over the dimension #{} "
                "(0-based indeces).".format(unknown_dim_index))

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
        self.initialize_variables(input)

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

            self.updates = [(
                self.running_inv_std,
                asfloat(1 - alpha) * self.running_inv_std + alpha * inv_std
            ), (
                self.running_mean,
                asfloat(1 - alpha) * self.running_mean + alpha * mean
            )]

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
        Coefficient, see equation above

    beta : float
        Offset, see equation above

    k : float
        Exponent, see equation above

    depth_radius : int
        Number of adjacent channels to normalize over, must be odd.

    {Identity.name}

    Methods
    -------
    {Identity.Methods}

    Attributes
    ----------
    {Identity.Attributes}
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

    def output(self, input):
        input = tf.convert_to_tensor(input, dtype=tf.float32)
        ndim = len(input.shape)

        if ndim != 3:
            raise LayerConnectionError(
                "Layer `{}` expected input with 3 dimensions, got {}"
                "".format(self.name, ndim))

        return tf.nn.local_response_normalization(
            input,
            depth_radius=self.depth_radius,
            bias=self.k,
            alpha=self.alpha,
            beta=self.beta)
