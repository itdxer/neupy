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
    WithdrawProperty, ParameterProperty,
)
from .base import BaseLayer


__all__ = ('Convolution', 'Deconvolution')


class Spatial2DProperty(TypedListProperty):
    """
    Stride property.

    Parameters
    ----------
    {BaseProperty.Parameters}
    """
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
                "Stide can have only one or two elements "
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
        raise ValueError("Deconvolutional layer doesn't support dilation")

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

    # We can think of the dilation as very sparse convolutional fitler
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
    """
    Border mode property identifies border for the
    convolution operation.
    Parameters
    ----------
    {Property.Parameters}
    """
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
        Rate for the fiter upsampling. When ``dilation > 1`` layer will
        become diated convolution (or atrous convolution). Defaults to ``1``.

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

    >>> from neupy import layers
    >>>
    >>> layers.join(
    ...     layers.Input((30, 10)),
    ...     layers.Reshape((30, 1, 10)),
    ...     layers.Convolution((3, 1, 16)),
    ... )

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

        if bias is not None:
            _, _, n_filters = size
            self.bias = self.variable(
                value=self.bias, name='bias',
                shape=as_tuple(n_filters))

    def fail_if_shape_invalid(self, input_shape):
        if input_shape and len(input_shape) != 3:
            raise LayerConnectionError(
                "Convolutional layer expects an input with 3 "
                "dimensions, got {} with shape {}"
                "".format(len(input_shape), input_shape))

    def output_shape_per_dim(self, *args, **kwargs):
        return conv_output_shape(*args, **kwargs)

    def get_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)

        if input_shape.ndims is None:
            return tf.TensorShape((None, None, None))

        self.fail_if_shape_invalid(input_shape)

        padding = self.padding
        rows, cols, _ = input_shape

        row_filter_size, col_filter_size, n_kernels = self.size
        row_stride, col_stride = self.stride
        row_dilation, col_dilation = self.dilation or (1, 1)

        if isinstance(padding, (list, tuple)):
            row_padding, col_padding = padding
        else:
            row_padding, col_padding = padding, padding

        output_rows = self.output_shape_per_dim(
            rows, row_filter_size,
            row_padding, row_stride, row_dilation,
        )
        output_cols = self.output_shape_per_dim(
            cols, col_filter_size,
            col_padding, col_stride, col_dilation,
        )

        return tf.TensorShape((output_rows, output_cols, n_kernels))

    def output(self, input_value, **kwargs):
        input_value = tf.convert_to_tensor(input_value, tf.float32)
        input_shape = input_value.shape
        padding = self.padding

        self.fail_if_shape_invalid(input_shape[1:])
        n_channels = input_shape[-1]
        n_rows, n_cols, n_filters = self.size

        self.weight = self.variable(
            value=self.weight, name='weight',
            shape=(n_rows, n_cols, n_channels, n_filters))

        if not isinstance(padding, six.string_types):
            height_pad, weight_pad = padding
            input_value = tf.pad(input_value, [
                [0, 0],
                [height_pad, height_pad],
                [weight_pad, weight_pad],
                [0, 0],
            ])
            # VALID option will make sure that
            # convolution won't use any padding.
            padding = 'VALID'

        output = tf.nn.convolution(
            input_value,
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
        classname = self.__class__.__name__
        return '{name}({size})'.format(name=classname, size=self.size)


class Deconvolution(Convolution):
    """
    Deconvolution layer. It's commonly called like this in the literature,
    but it's just gradient of the convolution and not actual deconvolution.

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

    Examples
    --------
    >>> from neupy import layers
    >>>
    >>> layers.join(
    ...     layers.Input((28, 28, 3)),
    ...     layers.Convolution((3, 3, 16)),
    ...     layers.Deconvolution((3, 3, 1)),
    ... )

    Attributes
    ----------
    {Convolution.Attributes}
    """
    def __init__(self, size, padding='valid', stride=1,
                 weight=init.HeNormal(gain=2), bias=0, name=None):

        super(Deconvolution, self).__init__(
            size=size, padding=padding, stride=stride,
            dilation=1, weight=weight, bias=bias, name=name)

    def output_shape_per_dim(self, *args, **kwargs):
        return deconv_output_shape(*args, **kwargs)

    @property
    def weight_shape(self):
        return as_tuple(self.size, self.input_shape[-1])

    def output(self, input_value, **kwargs):
        input_value = tf.convert_to_tensor(input_value, tf.float32)
        input_shape = input_value.shape
        padding = self.padding
        batch_size = input_shape[0]

        self.fail_if_shape_invalid(input_shape[1:])
        n_channels = input_shape[-1]
        n_rows, n_cols, n_filters = self.size

        # We need to get information about output shape from the input
        # tensor's shape, because for some inputs we might have
        # height and width specified as None and shape value won't be
        # computed for these dimensions.
        output_shape = self.get_output_shape(
            tf.unstack(input_shape[1:]))

        # Compare to the regular convolution weights,
        # transposed one has switched input and output channels.
        self.weight = self.variable(
            value=self.weight, name='weight',
            shape=(n_rows, n_cols, n_filters, n_channels))

        if isinstance(self.padding, (list, tuple)):
            height_pad, width_pad = self.padding

            # VALID option will make sure that
            # deconvolution won't use any padding.
            padding = 'VALID'

            # conv2d_transpose doesn't know about extra paddings that we added
            # in the convolution. For this reason we have to expand our
            # expected output shape and later we will remove these paddings
            # manually after transpose convolution.
            output_shape = (
                output_shape[0] + 2 * height_pad,
                output_shape[1] + 2 * width_pad,
                output_shape[2],
            )

        output = tf.nn.conv2d_transpose(
            input_value,
            self.weight,
            as_tuple(batch_size, output_shape),
            as_tuple(1, self.stride, 1),
            padding,
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
