from __future__ import division

import math
import collections

import six
import tensorflow as tf

from neupy.utils import as_tuple
from neupy.exceptions import LayerConnectionError
from neupy.core.properties import TypedListProperty, Property
from .base import ParameterBasedLayer


__all__ = ('Convolution', 'Deconvolution')


class StrideProperty(TypedListProperty):
    """
    Stride property.

    Parameters
    ----------
    {BaseProperty.Parameters}
    """
    expected_type = (list, tuple, int)

    def __init__(self, *args, **kwargs):
        kwargs['element_type'] = int
        super(StrideProperty, self).__init__(*args, **kwargs)

    def __set__(self, instance, value):
        if isinstance(value, collections.Iterable) and len(value) == 1:
            value = (value[0], 1)

        if isinstance(value, int):
            value = (value, value)

        super(StrideProperty, self).__set__(instance, value)

    def validate(self, value):
        super(StrideProperty, self).validate(value)

        if len(value) > 2:
            raise ValueError("Stide can have only one or two elements "
                             "in the list. Got {}".format(len(value)))

        if any(element <= 0 for element in value):
            raise ValueError(
                "Stride size should contain only values greater than zero")


def deconv_output_shape(dimension_size, filter_size, padding, stride):
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

    Returns
    -------
    int
        Dimension size after applying deconvolution
        operation with specified configurations.
    """
    if dimension_size is None:
        return None

    if padding in ('VALID', 'valid'):
        return dimension_size * stride + max(filter_size - stride, 0)

    elif padding in ('SAME', 'same'):
        return dimension_size * stride

    elif isinstance(padding, int):
        return dimension_size * stride - 2 * padding + filter_size - 1

    raise ValueError("`{!r}` is unknown deconvolution's padding value"
                     "".format(padding))


def conv_output_shape(dimension_size, filter_size, padding, stride):
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

    Returns
    -------
    int
        Dimension size after applying convolution
        operation with specified configurations.
    """
    if dimension_size is None:
        return None

    if not isinstance(stride, int):
        raise ValueError("Stride needs to be an integer, got {} (value {!r})"
                         "".format(type(stride), stride))

    if not isinstance(filter_size, int):
        raise ValueError("Filter size needs to be an integer, got {} "
                         "(value {!r})".format(type(filter_size),
                                               filter_size))

    if padding in ('VALID', 'valid'):
        return math.ceil((dimension_size - filter_size + 1) / stride)

    elif padding in ('SAME', 'same'):
        return math.ceil(dimension_size / stride)

    elif isinstance(padding, int):
        return math.ceil(
            (dimension_size + 2 * padding - filter_size + 1) / stride)

    raise ValueError("`{!r}` is unknown convolution's padding value"
                     "".format(padding))


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
                    "greater or equal to zero, got {}".format(value)
                )

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
                "".format(len(value))
            )

        is_invalid_string = (
            isinstance(value, six.string_types) and
            value not in self.valid_string_choices
        )

        if is_invalid_string:
            valid_choices = ', '.join(self.valid_string_choices)
            raise ValueError("`{}` is invalid string value. Available: {}"
                             "".format(value, valid_choices))

        if isinstance(value, tuple) and any(element < 0 for element in value):
            raise ValueError("Tuple border mode value needs to contain "
                             "only elements that greater or equal to zero, "
                             "got {}".format(value))


class Convolution(ParameterBasedLayer):
    """
    Convolutional layer.

    Parameters
    ----------
    size : tuple of int
        Filter shape. In should be defined as a tuple with three
        integers ``(filter rows, filter columns, output channels)``.

    padding : {{``VALID``, ``SAME``}} or int
        Defaults to ``VALID``.

    stride : tuple with ints, int.
        Stride size. Defaults to ``(1, 1)``

    weight : array-like, Tensorfow variable, scalar or Initializer
        Defines layer's weights. Shape of the weight will be equal to
        ``(filter rows, filter columns, input channels, output channels)``.
        Default initialization methods you can find
        :ref:`here <init-methods>`. Defaults to
        :class:`XavierNormal() <neupy.init.XavierNormal>`.

    {ParameterBasedLayer.bias}

    {BaseLayer.Parameters}

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
    {ParameterBasedLayer.Methods}

    Attributes
    ----------
    {ParameterBasedLayer.Attributes}
    """
    size = TypedListProperty(required=True, element_type=int)
    padding = PaddingProperty(default='valid')
    stride = StrideProperty(default=(1, 1))

    def validate(self, input_shape):
        if len(input_shape) != 3:
            raise LayerConnectionError(
                "Convolutional layer expects an input with 3 "
                "dimensions, got {} with shape {}"
                "".format(len(input_shape), input_shape))

    def output_shape_per_dim(self, *args, **kwargs):
        return conv_output_shape(*args, **kwargs)

    @property
    def output_shape(self):
        if self.input_shape is None:
            return None

        padding = self.padding
        rows, cols, _ = self.input_shape
        row_filter_size, col_filter_size, n_kernels = self.size

        row_stride, col_stride = self.stride

        if isinstance(padding, (list, tuple)):
            row_padding, col_padding = padding
        else:
            row_padding, col_padding = padding, padding

        output_rows = self.output_shape_per_dim(
            rows, row_filter_size, row_padding, row_stride)

        output_cols = self.output_shape_per_dim(
            cols, col_filter_size, col_padding, col_stride)

        # In python 2, we can get float number after rounding procedure
        # and it might break processing in the subsequent layers.
        return (int(output_rows), int(output_cols), n_kernels)

    @property
    def weight_shape(self):
        n_channels = self.input_shape[-1]
        n_rows, n_cols, n_filters = self.size
        return (n_rows, n_cols, n_channels, n_filters)

    @property
    def bias_shape(self):
        return as_tuple(self.size[-1])

    def output(self, input_value):
        padding = self.padding

        if not isinstance(padding, six.string_types):
            height_pad, weight_pad = padding
            input_value = tf.pad(input_value, [
                [0, 0],
                [height_pad, height_pad],
                [weight_pad, weight_pad],
                [0, 0],
            ])
            # We will need to make sure that convolution operation
            # won't add any paddings.
            padding = 'VALID'

        output = tf.nn.convolution(
            input_value,
            self.weight,
            padding,
            self.stride,
            data_format="NHWC"
        )

        if self.bias is not None:
            bias = tf.reshape(self.bias, (1, 1, 1, -1))
            output += bias

        return output


class Deconvolution(Convolution):
    """
    Deconvolution layer. It's commonly called like this in the literature,
    but it's just gradient of the convolution and not actual deconvolution.

    Parameters
    ----------
    {Convolution.Parameters}

    Methods
    -------
    {ParameterBasedLayer.Methods}

    Attributes
    ----------
    {ParameterBasedLayer.Attributes}
    """
    def output_shape_per_dim(self, *args, **kwargs):
        return deconv_output_shape(*args, **kwargs)

    @property
    def weight_shape(self):
        return as_tuple(self.size, self.input_shape[-1])

    def output(self, input_value):
        input_shape = tf.shape(input_value)
        output_shape = self.output_shape

        batch_size = input_shape[0]
        padding = self.padding

        if isinstance(self.padding, (list, tuple)):
            height_pad, width_pad = self.padding
            padding = 'VALID'

            # conv2d transpose doesn't know about extra paddings that we added
            # in the convolution. For this reason we have to expand our
            # expected output shape and later we will remove these paddings
            # manually
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
            height_pad, width_pad = self.padding
            output = output[
                :, height_pad:-height_pad, width_pad:-width_pad, :]

        if self.bias is not None:
            bias = tf.reshape(self.bias, (1, 1, 1, -1))
            output += bias

        return output
