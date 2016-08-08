import collections

import six
import theano.tensor as T
from theano.tensor.signal import pool

from neupy.utils import as_tuple
from neupy.core.properties import TypedListProperty, Property, ChoiceProperty
from .connections import LayerConnectionError
from .base import BaseLayer, ParameterBasedLayer


__all__ = ('Convolution', 'MaxPooling', 'AveragePooling', 'Upscale')


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


def conv_output_shape(dimension_size, filter_size, border_mode, stride):
    """
    Computes convolution's output shape.

    Parameters
    ----------
    dimension_size : int
    filter_size : int
    border_mode : {'valid', 'full', 'half'} or int
    stride : int

    Returns
    -------
    int
    """
    if not isinstance(stride, int):
        raise ValueError("Stride needs to be an integer, got {} (value {!r})"
                         "".format(type(stride), stride))

    if not isinstance(filter_size, int):
        raise ValueError("Filter size needs to be an integer, got {} "
                         "(value {!r})".format(type(filter_size),
                                               filter_size))

    if border_mode == 'valid':
        output_size = dimension_size - filter_size + 1

    elif border_mode == 'half':
        output_size = dimension_size + 2 * (filter_size // 2) - filter_size + 1

    elif border_mode == 'full':
        output_size = dimension_size + filter_size - 1

    elif isinstance(border_mode, int):
        output_size = dimension_size + 2 * border_mode - filter_size + 1

    else:
        raise ValueError("`{!r}` is unknown convolution's border mode value"
                         "".format(border_mode))

    return (output_size + stride - 1) // stride


class Convolution(ParameterBasedLayer):
    """
    Convolutional layer.

    Parameters
    ----------
    size : tuple of int
        Filter shape. In should be defined as a tuple with three integers
        ``(output channels, filter rows, filter columns)``.
    border_mode : {{'valid', 'full', 'half'}} or int or tuple with 2 int
        Convolution border mode. Check Theano's ``nnet.conv2d`` doc.
    stride_size : tuple with 1 or 2 integers or integer.
        Stride size.
    {ParameterBasedLayer.weight}
    {ParameterBasedLayer.bias}

    Methods
    -------
    {ParameterBasedLayer.Methods}

    Attributes
    ----------
    {ParameterBasedLayer.Attributes}
    """
    size = TypedListProperty(required=True, element_type=int)
    border_mode = BorderModeProperty(default='valid')
    stride_size = StrideProperty(default=(1, 1))

    @property
    def output_shape(self):
        if self.input_shape is None:
            return None

        if len(self.input_shape) < 2:
            raise ValueError(
                "Convolutional layer expects an input shape with least 2 "
                "dimensions, got {} with shape {}".format(
                    len(self.input_shape),
                    self.input_shape
                )
            )

        border_mode = self.border_mode
        n_kernels = self.size[0]
        rows, cols = self.input_shape[-2:]
        row_filter_size, col_filter_size = self.size[-2:]
        row_stride, col_stride = self.stride_size

        if isinstance(border_mode, tuple):
            row_border_mode, col_border_mode = border_mode[-2:]
        else:
            row_border_mode, col_border_mode = border_mode, border_mode

        output_rows = conv_output_shape(rows, row_filter_size,
                                        row_border_mode, row_stride)
        output_cols = conv_output_shape(cols, col_filter_size,
                                        col_border_mode, col_stride)
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
        bias = T.reshape(self.bias, (1, -1, 1, 1))
        output = T.nnet.conv2d(input_value, self.weight,
                               input_shape=as_tuple(None, self.input_shape),
                               filter_shape=self.weight_shape,
                               border_mode=self.border_mode,
                               subsample=self.stride_size)
        return output + bias


class BasePooling(BaseLayer):
    """
    Base class for the pooling layers.

    Parameters
    ----------
    size : tuple with 2 integers
        Factor by which to downscale (vertical, horizontal).
        (2, 2) will halve the image in each dimension.
    stride_size : tuple with 1 or 2 integers or integer.
        Stride size, which is the number of shifts over
        rows/cols to get the next pool region. If stride_size is
        None, it is considered equal to ds (no overlap on
        pooling regions).
    padding : tuple of two ints
        (pad_h, pad_w), pad zeros to extend beyond four borders of
        the images, pad_h is the size of the top and bottom margins,
        and pad_w is the size of the left and right margins.

    Methods
    -------
    {BaseLayer.Methods}

    Attributes
    ----------
    {BaseLayer.Attributes}
    """
    size = TypedListProperty(required=True, element_type=int)
    stride_size = StrideProperty(default=None)
    padding = TypedListProperty(default=(0, 0), element_type=int, n_elements=2)

    def __init__(self, size, **options):
        super(BasePooling, self).__init__(size=size, **options)

    @property
    def output_shape(self):
        if self.input_shape is None:
            return None

        if len(self.input_shape) < 3:
            raise ValueError(
                "Convolutional layer expects an input shape with least 3 "
                "dimensions, got {} with shape {}".format(
                    len(self.input_shape),
                    self.input_shape
                )
            )

        n_kernels, rows, cols = self.input_shape[-3:]
        row_filter_size, col_filter_size = self.size

        stride_size = self.stride_size
        if stride_size is None:
            stride_size = self.size

        row_stride, col_stride = stride_size
        row_border_mode, col_border_mode = self.padding

        output_rows = conv_output_shape(rows, row_filter_size,
                                        row_border_mode, row_stride)
        output_cols = conv_output_shape(cols, col_filter_size,
                                        col_border_mode, col_stride)
        return (n_kernels, output_rows, output_cols)

    def __repr__(self):
        return '{name}({size})'.format(name=self.__class__.__name__,
                                       size=self.size)


class MaxPooling(BasePooling):
    """
    Maximum pooling layer.

    Parameters
    ----------
    {BasePooling.Parameters}

    Methods
    -------
    {BasePooling.Methods}

    Attributes
    ----------
    {BasePooling.Attributes}
    """
    def output(self, input_value):
        return pool.pool_2d(input_value, ds=self.size, mode='max',
                            ignore_border=True, st=self.stride_size,
                            padding=self.padding)


class AveragePooling(BasePooling):
    """
    Average pooling layer.

    Parameters
    ----------
    mode : {{'include_padding', 'exclude_padding'}}
        Gives you the choice to include or exclude padding.
        Defaults to ``include_padding``.
    {BasePooling.Parameters}

    Methods
    -------
    {BasePooling.Methods}

    Attributes
    ----------
    {BasePooling.Attributes}
    """
    mode = ChoiceProperty(
        default='include_padding',
        choices={
            'include_padding': 'average_inc_pad',
            'exclude_padding': 'average_exc_pad'
        }
    )

    def output(self, input_value):
        return pool.pool_2d(input_value, ds=self.size, mode=self.mode,
                            ignore_border=True, st=self.stride_size,
                            padding=self.padding)


class ScaleFactorProperty(TypedListProperty):
    """
    Defines sclaing factor for the Upscale layer.

    Parameters
    ----------
    {TypedListProperty.Parameters}
    """
    expected_type = (tuple, int)

    def __set__(self, instance, value):
        if isinstance(value, int):
            value = as_tuple(value, value)
        super(ScaleFactorProperty, self).__set__(instance, value)

    def validate(self, value):
        if any(element <= 0 for element in value):
            raise ValueError("Scale factor property accepts only positive "
                             "integer numbers.")
        super(ScaleFactorProperty, self).validate(value)


class Upscale(BaseLayer):
    """
    Upscales input over two axis (height and width).

    Parameters
    ----------
    scale : int or tuple with two int
        Scaling factor for the input value. In the tuple first
        parameter identifies scale of the height and the second
        one of the width.

    Methods
    -------
    {BaseLayer.Methods}

    Attributes
    ----------
    {BaseLayer.Attributes}
    """
    scale = ScaleFactorProperty(required=True, n_elements=2)

    def __init__(self, scale, **options):
        super(Upscale, self).__init__(scale=scale, **options)

    @property
    def output_shape(self):
        if len(self.input_shape) != 3:
            raise LayerConnectionError(
                "Upscale layer should have an input value that have "
                "3 feature dimensions (channel, height and width)"
            )

        channel, height, width = self.input_shape
        height_scale, width_scale = self.scale

        return (channel, height_scale * height, width_scale * width)

    def output(self, input_value):
        height_scale, width_scale = self.scale
        scaled_value = input_value

        if height_scale != 1:
            scaled_value = T.extra_ops.repeat(scaled_value, height_scale, 2)

        if width_scale != 1:
            scaled_value = T.extra_ops.repeat(scaled_value, width_scale, 3)

        return scaled_value
