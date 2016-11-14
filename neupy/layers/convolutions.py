import collections

import six
import theano.tensor as T

from neupy.utils import as_tuple
from neupy.core.properties import TypedListProperty, Property
from .base import ParameterBasedLayer
from .connections import LayerConnectionError


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


class BorderModeProperty(Property):
    """
    Border mode property identifies border for the
    convolution operation.

    Parameters
    ----------
    {Property.Parameters}
    """
    expected_type = (six.string_types, int, tuple)
    valid_string_choices = ('valid', 'full', 'half')

    def validate(self, value):
        super(BorderModeProperty, self).validate(value)

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

        if isinstance(value, int) and value < 0:
            raise ValueError("Integer border mode value needs to be "
                             "greater or equal to zero, got {}".format(value))

        if isinstance(value, tuple) and any(element < 0 for element in value):
            raise ValueError("Tuple border mode value needs to contain "
                             "only elements that greater or equal to zero, "
                             "got {}".format(value))


def conv_output_shape(dimension_size, filter_size, padding, stride):
    """
    Computes convolution's output shape.

    Parameters
    ----------
    dimension_size : int
    filter_size : int
    padding : {'valid', 'full', 'half'} or int
    stride : int

    Returns
    -------
    int
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

    if padding == 'valid':
        output_size = dimension_size - filter_size + 1

    elif padding == 'half':
        output_size = dimension_size + 2 * (filter_size // 2) - filter_size + 1

    elif padding == 'full':
        output_size = dimension_size + filter_size - 1

    elif isinstance(padding, int):
        output_size = dimension_size + 2 * padding - filter_size + 1

    else:
        raise ValueError("`{!r}` is unknown convolution's border mode value"
                         "".format(padding))

    return (output_size + stride - 1) // stride


class Convolution(ParameterBasedLayer):
    """
    Convolutional layer.

    Parameters
    ----------
    size : tuple of int
        Filter shape. In should be defined as a tuple with three integers
        ``(output channels, filter rows, filter columns)``.
    padding : {{'valid', 'full', 'half'}} or int or tuple with 2 int
        Convolution border mode. Check Theano's ``nnet.conv2d`` doc.
    stride : tuple with 1 or 2 integers or integer.
        Stride size.
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
    size = TypedListProperty(required=True, element_type=int)
    padding = BorderModeProperty(default='valid')
    stride = StrideProperty(default=(1, 1))

    @property
    def output_shape(self):
        if self.input_shape is None:
            return None

        if len(self.input_shape) < 2:
            raise LayerConnectionError(
                "Convolutional layer expects an input with at least 2 "
                "dimensions, got {} with shape {}"
                "".format(len(self.input_shape), self.input_shape)
            )

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
        return (n_filters, n_channels, n_rows, n_cols)

    @property
    def bias_shape(self):
        return as_tuple(self.size[0])

    def output(self, input_value):
        output = T.nnet.conv2d(input_value, self.weight,
                               input_shape=as_tuple(None, self.input_shape),
                               filter_shape=self.weight_shape,
                               border_mode=self.padding,
                               subsample=self.stride)

        if self.bias is not None:
            bias = T.reshape(self.bias, (1, -1, 1, 1))
            output += bias

        return output
