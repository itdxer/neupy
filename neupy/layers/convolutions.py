import math
import collections

import six
import tensorflow as tf

from neupy.utils import as_tuple
from neupy.exceptions import LayerConnectionError
from neupy.core.properties import TypedListProperty, Property, ChoiceProperty
from .base import ParameterBasedLayer


__all__ = ('Convolution',)


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
            raise ValueError("Stride size should contain only values greater "
                             "than zero")


def conv_output_shape(dimension_size, filter_size, padding, stride):
    """
    Computes convolution's output shape.

    Parameters
    ----------
    dimension_size : int
        Size of the dimension. Typically it's image's
        weight or height.

    filter_size : int
        Size of the convolution filter.

    padding : {``valid``, ``full``, ``half``} or int
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

    if padding == 'VALID':
        return math.ceil((dimension_size - filter_size + 1) / stride)

    elif padding == 'SAME':
        return math.ceil(dimension_size / stride)

    raise ValueError("`{!r}` is unknown convolution's padding value"
                     "".format(padding))


class Convolution(ParameterBasedLayer):
    """
    Convolutional layer.

    Parameters
    ----------
    size : tuple of int
        Filter shape. In should be defined as a tuple with three integers
        ``(output channels, filter rows, filter columns)``.

    padding : {{``valid``, ``full``, ``half``}} or int or tuple with 2 int
        Convolution border mode. Check Theano's ``nnet.conv2d`` doc.
        Defaults to ``valid``.

    stride : tuple with 1 or 2 integers or integer.
        Stride size. Defaults to ``(1, 1)``

    {ParameterBasedLayer.weight}

    {ParameterBasedLayer.bias}

    {BaseLayer.Parameters}

    Examples
    --------
    2D Convolution

    >>> from neupy import layers
    >>>
    >>> layers.join(
    ...     layers.Input((3, 28, 28)),
    ...     layers.Convolution((16, 3, 3)),
    ... )

    1D Convolution

    >>> from neupy import layers
    >>>
    >>> layers.join(
    ...     layers.Input((10, 30)),
    ...     layers.Reshape((10, 30, 1)),
    ...     layers.Convolution((16, 3, 1)),
    ... )

    Methods
    -------
    {ParameterBasedLayer.Methods}

    Attributes
    ----------
    {ParameterBasedLayer.Attributes}
    """
    size = TypedListProperty(required=True, element_type=int)
    padding = ChoiceProperty(default='VALID', choices=('VALID', 'SAME'))
    stride = StrideProperty(default=(1, 1))

    def validate(self, input_shape):
        if len(input_shape) != 3:
            raise LayerConnectionError(
                "Convolutional layer expects an input with 3 "
                "dimensions, got {} with shape {}"
                "".format(len(input_shape), input_shape))

    @property
    def output_shape(self):
        if self.input_shape is None:
            return None

        padding = self.padding
        n_kernels = self.size[0]
        rows, cols = self.input_shape[-2:]
        row_filter_size, col_filter_size = self.size[-2:]
        row_stride, col_stride = self.stride

        if isinstance(padding, tuple):
            row_padding, col_padding = padding[-2:]
        else:
            row_padding, col_padding = padding, padding

        output_rows = conv_output_shape(rows, row_filter_size,
                                        row_padding, row_stride)
        output_cols = conv_output_shape(cols, col_filter_size,
                                        col_padding, col_stride)
        return (n_kernels, output_rows, output_cols)

    @property
    def weight_shape(self):
        n_channels = self.input_shape[0]
        n_filters, n_rows, n_cols = self.size
        return (n_rows, n_cols, n_channels, n_filters)

    @property
    def bias_shape(self):
        return as_tuple(self.size[0])

    def output(self, input_value):
        # TODO: transpose added only for convenient transition between
        # tensroflow and theatno. I will remove it later.
        input_value = tf.transpose(input_value, (0, 2, 3, 1))
        output = tf.nn.convolution(
            input_value, self.weight,
            self.padding, self.stride,
        )
        output = tf.transpose(output, (0, 3, 1, 2))

        if self.bias is not None:
            bias = tf.reshape(self.bias, (1, -1, 1, 1))
            output += bias

        return output
