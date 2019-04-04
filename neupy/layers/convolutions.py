from __future__ import division

import math
import collections

import six
import tensorflow as tf

from neupy import init
from neupy.utils import as_tuple
from neupy.exceptions import LayerConnectionError
from neupy.core.properties import (
    TypedListProperty, Property,
    ParameterProperty,
)
from .base import BaseLayer


__all__ = ('Convolution', 'Deconvolution')


class Spatial2DProperty(TypedListProperty):
    expected_type = (list, tuple, int)

    def __init__(self, *args, **kwargs):
        kwargs['element_type'] = int
        super(Spatial2DProperty, self).__init__(*args, **kwargs)

    def __set__(self, instance, value):
        if isinstance(value, collections.Iterable) and len(value) == 1:
            value = (value[0], 1)

        if isinstance(value, int):
            value = (value, value)

        super(Spatial2DProperty, self).__set__(instance, value)

    def validate(self, value):
        super(Spatial2DProperty, self).validate(value)

        if len(value) > 2:
            raise ValueError(
                "Stride can have only one or two elements "
                "in the list. Got {}".format(len(value)))

        if any(element <= 0 for element in value):
            raise ValueError(
                "Stride size should contain only values greater than zero")


def deconv_output_shape(dimension_size, filter_size, padding, stride,
                        dilation=1):
    """
    Computes deconvolution's output shape for one spatial dimension.

    Parameters
    ----------
    dimension_size : int or None
        Size of the dimension. Typically it's image's weight or height.
        It might be equal to ``None`` when we input might have variable
        dimension.

    filter_size : int
        Size of the convolution filter.

    padding : {``valid``, ``same``} or int
        Type or size of the zero-padding.

    stride : int
        Stride size.

    dilation : int
        Dilation rate. Only ``dilation=1`` is supported for the
        deconvolution.

    Returns
    -------
    int
        Dimension size after applying deconvolution
        operation with specified configurations.
    """
    if isinstance(dimension_size, tf.Dimension):
        dimension_size = dimension_size.value

    if dimension_size is None:
        return None

    if dilation != 1:
        raise ValueError("Deconvolution layer doesn't support dilation")

    if padding in ('VALID', 'valid'):
        return dimension_size * stride + max(filter_size - stride, 0)

    elif padding in ('SAME', 'same'):
        return dimension_size * stride

    elif isinstance(padding, int):
        return dimension_size * stride - 2 * padding + filter_size - 1

    raise ValueError(
        "`{!r}` is unknown deconvolution's padding value".format(padding))


def conv_output_shape(dimension_size, filter_size, padding, stride,
                      dilation=1):
    """
    Computes convolution's output shape for one spatial dimension.

    Parameters
    ----------
    dimension_size : int or None
        Size of the dimension. Typically it's image's weight or height.
        It might be equal to ``None`` when we input might have variable
        dimension.

    filter_size : int
        Size of the convolution filter.

    padding : {``valid``, ``same``} or int
        Type or size of the zero-padding.

    stride : int
        Stride size.

    dilation : int
        Dilation rate. Defaults to ``1``.

    Returns
    -------
    int
        Dimension size after applying convolution
        operation with specified configurations.
    """
    if isinstance(dimension_size, tf.Dimension):
        dimension_size = dimension_size.value

    if dimension_size is None:
        return None

    if not isinstance(stride, int):
        raise ValueError(
            "Stride needs to be an integer, got {} (value {!r})"
            "".format(type(stride), stride))

    if not isinstance(filter_size, int):
        raise ValueError(
            "Filter size needs to be an integer, got {} "
            "(value {!r})".format(type(filter_size), filter_size))

    # We can think of the dilation as very sparse convolutional filter
    # filter=3 and dilation=2 the same as filter=5 and dilation=1
    filter_size = filter_size + (filter_size - 1) * (dilation - 1)

    if padding in ('VALID', 'valid'):
        return int(math.ceil((dimension_size - filter_size + 1) / stride))

    elif padding in ('SAME', 'same'):
        return int(math.ceil(dimension_size / stride))

    elif isinstance(padding, int):
        return int(math.ceil(
            (dimension_size + 2 * padding - filter_size + 1) / stride))

    raise ValueError(
        "`{!r}` is unknown convolution's padding value".format(padding))


class PaddingProperty(Property):
    expected_type = (six.string_types, int, tuple)
    valid_string_choices = ('VALID', 'SAME', 'same', 'valid')

    def __set__(self, instance, value):
        if isinstance(value, int):
            if value < 0:
                raise ValueError(
                    "Integer border mode value needs to be "
                    "greater or equal to zero, got {}".format(value))

            value = (value, value)

        if isinstance(value, six.string_types):
            value = value.upper()

        super(PaddingProperty, self).__set__(instance, value)

    def validate(self, value):
        super(PaddingProperty, self).validate(value)

        if isinstance(value, tuple) and len(value) != 2:
            raise ValueError(
                "Border mode property suppose to get a tuple that "
                "contains two elements, got {} elements"
                "".format(len(value)))

        is_invalid_string = (
            isinstance(value, six.string_types) and
            value not in self.valid_string_choices
        )

        if is_invalid_string:
            valid_choices = ', '.join(self.valid_string_choices)
            raise ValueError(
                "`{}` is invalid string value. Available choices: {}"
                "".format(value, valid_choices))

        if isinstance(value, tuple) and any(element < 0 for element in value):
            raise ValueError(
                "Tuple border mode value needs to contain only elements "
                "that greater or equal to zero, got {}".format(value))


class Convolution(BaseLayer):
    """
    Convolutional layer.

    Parameters
    ----------
    size : tuple of int
        Filter shape. In should be defined as a tuple with three
        integers ``(filter rows, filter columns, output channels)``.

    padding : {{``same``, ``valid``}}, int, tuple
        Zero padding for the input tensor.

        - ``valid`` - Padding won't be added to the tensor. Result will be
          the same as for ``padding=0``

        - ``same`` - Padding will depend on the number of rows and columns
          in the filter. This padding makes sure that image with the
          ``stride=1`` won't change its width and height. It's the same as
          ``padding=(filter rows // 2, filter columns // 2)``.

        - Custom value for the padding can be specified as an integer, like
          ``padding=1`` or it can be specified as a tuple when different
          dimensions have different padding values, for example
          ``padding=(2, 3)``.

        Defaults to ``valid``.

    stride : tuple with ints, int.
        Stride size. Defaults to ``(1, 1)``

    dilation : int, tuple
        Rate for the filter upsampling. When ``dilation > 1`` layer will
        become dilated convolution (or atrous convolution). Defaults to ``1``.

    weight : array-like, Tensorfow variable, scalar or Initializer
        Defines layer's weights. Shape of the weight will be equal to
        ``(filter rows, filter columns, input channels, output channels)``.
        Default initialization methods you can find
        :ref:`here <init-methods>`. Defaults to
        :class:`HeNormal(gain=2) <neupy.init.HeNormal>`.

    bias : 1D array-like, Tensorfow variable, scalar, Initializer or None
        Defines layer's bias. Default initialization methods you can find
        :ref:`here <init-methods>`. Defaults to
        :class:`Constant(0) <neupy.init.Constant>`.
        The ``None`` value excludes bias from the calculations and
        do not add it into parameters list.

    {BaseLayer.name}

    Examples
    --------
    2D Convolution

    >>> from neupy import layers
    >>>
    >>> layers.join(
    ...     layers.Input((28, 28, 3)),
    ...     layers.Convolution((3, 3, 16)),
    ... )

    1D Convolution

    >>> from neupy.layers import *
    >>> network = join(
    ...     Input((30, 10)),
    ...     Reshape((30, 1, 10)),  # convert 3D to 4D
    ...     Convolution((3, 1, 16)),
    ...     Reshape((-1, 16))  # convert 4D back to 3D
    ... )
    >>> network
    (?, 30, 10) -> [... 4 layers ...] -> (?, 28, 16)

    Methods
    -------
    {BaseLayer.Methods}

    Attributes
    ----------
    {BaseLayer.Attributes}
    """
    size = TypedListProperty(element_type=int, n_elements=3)
    weight = ParameterProperty()
    bias = ParameterProperty(allow_none=True)

    padding = PaddingProperty()
    stride = Spatial2DProperty()
    dilation = Spatial2DProperty()

    # We use gain=2 because it's suitable choice for relu non-linearity
    # and relu is the most common non-linearity used for CNN.
    def __init__(self, size, padding='valid', stride=1, dilation=1,
                 weight=init.HeNormal(gain=2), bias=0, name=None):

        super(Convolution, self).__init__(name=name)

        self.size = size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.weight = weight
        self.bias = bias

    def fail_if_shape_invalid(self, input_shape):
        if input_shape and input_shape.ndims != 4:
            raise LayerConnectionError(
                "Convolutional layer expects an input with 4 "
                "dimensions, got {} with shape {}"
                "".format(len(input_shape), input_shape))

    def output_shape_per_dim(self, *args, **kwargs):
        return conv_output_shape(*args, **kwargs)

    def expected_output_shape(self, input_shape):
        n_samples = input_shape[0]
        row_filter_size, col_filter_size, n_kernels = self.size
        row_stride, col_stride = self.stride
        row_dilation, col_dilation = self.dilation

        if isinstance(self.padding, (list, tuple)):
            row_padding, col_padding = self.padding
        else:
            row_padding, col_padding = self.padding, self.padding

        return (
            n_samples,
            self.output_shape_per_dim(
                input_shape[1], row_filter_size,
                row_padding, row_stride, row_dilation
            ),
            self.output_shape_per_dim(
                input_shape[2], col_filter_size,
                col_padding, col_stride, col_dilation
            ),
            n_kernels,
        )

    def get_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self.fail_if_shape_invalid(input_shape)

        if input_shape.ndims is None:
            n_samples = input_shape[0]
            n_kernels = self.size[-1]
            return tf.TensorShape((n_samples, None, None, n_kernels))

        return tf.TensorShape(self.expected_output_shape(input_shape))

    def create_variables(self, input_shape):
        self.input_shape = input_shape
        n_channels = input_shape[-1]
        n_rows, n_cols, n_filters = self.size

        # Compare to the regular convolution weights,
        # transposed one has switched input and output channels.
        self.weight = self.variable(
            value=self.weight, name='weight',
            shape=(n_rows, n_cols, n_channels, n_filters))

        if self.bias is not None:
            self.bias = self.variable(
                value=self.bias, name='bias',
                shape=as_tuple(n_filters))

    def output(self, input, **kwargs):
        input = tf.convert_to_tensor(input, tf.float32)
        self.fail_if_shape_invalid(input.shape)
        padding = self.padding

        if not isinstance(padding, six.string_types):
            height_pad, width_pad = padding
            input = tf.pad(input, [
                [0, 0],
                [height_pad, height_pad],
                [width_pad, width_pad],
                [0, 0],
            ])
            # VALID option will make sure that
            # convolution won't use any padding.
            padding = 'VALID'

        output = tf.nn.convolution(
            input,
            self.weight,
            padding=padding,
            strides=self.stride,
            dilation_rate=self.dilation,
            data_format="NHWC",
        )

        if self.bias is not None:
            bias = tf.reshape(self.bias, (1, 1, 1, -1))
            output += bias

        return output

    def __repr__(self):
        return self._repr_arguments(
            self.size,
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilation,
            weight=self.weight,
            bias=self.bias,
            name=self.name,
        )


class Deconvolution(Convolution):
    """
    Deconvolution layer (also known as Transposed Convolution.).

    Parameters
    ----------
    {Convolution.size}

    {Convolution.padding}

    {Convolution.stride}

    {Convolution.dilation}

    weight : array-like, Tensorfow variable, scalar or Initializer
        Defines layer's weights. Shape of the weight will be equal to
        ``(filter rows, filter columns, output channels, input channels)``.
        Default initialization methods you can find
        :ref:`here <init-methods>`. Defaults to
        :class:`HeNormal(gain=2) <neupy.init.HeNormal>`.

    {Convolution.bias}

    {Convolution.name}

    Methods
    -------
    {Convolution.Methods}

    Attributes
    ----------
    {Convolution.Attributes}

    Examples
    --------
    >>> from neupy.layers import *
    >>> network = join(
    ...     Input((28, 28, 3)),
    ...     Convolution((3, 3, 16)),
    ...     Deconvolution((3, 3, 1)),
    ... )
    >>> network
    (?, 28, 28, 3) -> [... 3 layers ...] -> (?, 28, 28, 1)
    """
    def __init__(self, size, padding='valid', stride=1,
                 weight=init.HeNormal(gain=2), bias=0, name=None):

        super(Deconvolution, self).__init__(
            size=size, padding=padding, stride=stride,
            dilation=1, weight=weight, bias=bias, name=name)

    def output_shape_per_dim(self, *args, **kwargs):
        return deconv_output_shape(*args, **kwargs)

    def create_variables(self, input_shape):
        self.input_shape = input_shape
        n_channels = input_shape[-1]
        n_rows, n_cols, n_filters = self.size

        # Compare to the regular convolution weights,
        # transposed one has switched input and output channels.
        self.weight = self.variable(
            value=self.weight, name='weight',
            shape=(n_rows, n_cols, n_filters, n_channels))

        if self.bias is not None:
            self.bias = self.variable(
                value=self.bias, name='bias',
                shape=as_tuple(n_filters))

    def output(self, input, **kwargs):
        input = tf.convert_to_tensor(input, tf.float32)
        # We need to get information about output shape from the input
        # tensor's shape, because for some inputs we might have
        # height and width specified as None and shape value won't be
        # computed for these dimensions.
        padding = self.padding

        # It's important that expected output shape gets computed on then
        # Tensor (produced by tf.shape) rather than on TensorShape object.
        # Tensorflow cannot convert TensorShape object into Tensor and
        # it will cause an exception in the conv2d_transpose layer.
        output_shape = self.expected_output_shape(tf.shape(input))

        if isinstance(self.padding, (list, tuple)):
            height_pad, width_pad = self.padding

            # VALID option will make sure that
            # deconvolution won't use any padding.
            padding = 'VALID'

            # conv2d_transpose doesn't know about extra paddings that we added
            # in the convolution. For this reason, we have to expand our
            # expected output shape and later we will remove these paddings
            # manually after transpose convolution.
            output_shape = (
                output_shape[0],
                output_shape[1] + 2 * height_pad,
                output_shape[2] + 2 * width_pad,
                output_shape[3],
            )

        output = tf.nn.conv2d_transpose(
            value=input,
            filter=self.weight,
            output_shape=list(output_shape),
            strides=as_tuple(1, self.stride, 1),
            padding=padding,
            data_format="NHWC"
        )

        if isinstance(self.padding, (list, tuple)):
            h_pad, w_pad = self.padding

            if h_pad > 0:
                output = output[:, h_pad:-h_pad, :, :]

            if w_pad > 0:
                output = output[:, :, w_pad:-w_pad, :]

        if self.bias is not None:
            bias = tf.reshape(self.bias, (1, 1, 1, -1))
            output += bias

        return output

    def __repr__(self):
        return self._repr_arguments(
            self.size,
            padding=self.padding,
            stride=self.stride,
            weight=self.weight,
            bias=self.bias,
            name=self.name,
        )
