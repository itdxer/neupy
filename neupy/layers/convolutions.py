import collections

import six
import theano.tensor as T
from theano.tensor.signal import pool

from neupy.core.properties import TypedListProperty, Property, ChoiceProperty
from .base import BaseLayer, ParameterBasedLayer


__all__ = ('Convolution', 'MaxPooling', 'AveragePooling')


class StrideProperty(TypedListProperty):
    """ Stride property.

    Parameters
    ----------
    {BaseProperty.default}
    {BaseProperty.required}
    """
    expected_type = (list, tuple, set, int)

    def __init__(self, *args, **kwargs):
        kwargs['element_type'] = int
        super(StrideProperty, self).__init__(*args, **kwargs)

    def __set__(self, instance, value):
        if isinstance(value, collections.Iterable) and len(value) == 1:
            value = value[0]

        if isinstance(value, int):
            value = (value, 1)

        super(StrideProperty, self).__set__(instance, value)

    def validate(self, value):
        super(StrideProperty, self).validate(value)
        if len(value) > 2:
            raise ValueError("Stide can have only one or two elements "
                             "in the list. Got {}".format(len(value)))


class BorderModeProperty(Property):
    """ Border mode property identifies border for the
    convolution operation.
    """
    expected_type = (six.string_types, int, tuple)
    valid_string_choices = ('valid', 'full', 'half')

    def validate(self, value):
        super(BorderModeProperty, self).validate(value)

        if isinstance(value, tuple):
            if len(value) != 2:
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


class Convolution(ParameterBasedLayer):
    """ Convolutional layer.

    Parameters
    ----------
    size : tuple of integers
        Filter shape.
    border_mode : {{'valid', 'full', 'half'}} or int or tuple with 2 int
        Convolution border mode. Check Theano's `nnet.conv2d` doc.
    stride_size : tuple with 1 or 2 integers or integer.
        Stride size.
    """
    size = TypedListProperty(required=True, element_type=int)
    border_mode = BorderModeProperty(default='valid')
    stride_size = StrideProperty(default=(1, 1))

    def weight_shape(self):
        return self.size

    def bias_shape(self):
        return self.size[:1]

    def output(self, input_value):
        bias = T.reshape(self.bias, (1, -1, 1, 1))
        output = T.nnet.conv2d(input_value, self.weight,
                               border_mode=self.border_mode,
                               subsample=self.stride_size)
        return output + bias


class BasePooling(BaseLayer):
    """ Base class for the pooling layers.

    Parameters
    ----------
    size : tuple with 2 integers
        Factor by which to downscale (vertical, horizontal).
        (2,2) will halve the image in each dimension.
    stride_size :
        Stride size, which is the number of shifts over
        rows/cols to get the next pool region. If stride_size is
        None, it is considered equal to ds (no overlap on
        pooling regions).
    padding : tuple of two ints
        (pad_h, pad_w), pad zeros to extend beyond four borders of
        the images, pad_h is the size of the top and bottom margins,
        and pad_w is the size of the left and right margins.
    """
    size = TypedListProperty(required=True, element_type=int)
    stride_size = StrideProperty(default=None)
    padding = TypedListProperty(default=(0, 0), element_type=int, n_elements=2)

    def __init__(self, size, **options):
        options['size'] = size
        super(BasePooling, self).__init__(**options)

    def __repr__(self):
        return '{name}({size})'.format(name=self.__class__.__name__,
                                       size=self.size)


class MaxPooling(BasePooling):
    """ Maximum pooling layer.

    Parameters
    ----------
    {BasePooling.size}
    {BasePooling.stride_size}
    """
    def output(self, input_value):
        return pool.pool_2d(input_value, ds=self.size, mode='max',
                            ignore_border=True, st=self.stride_size,
                            padding=self.padding)


class AveragePooling(BasePooling):
    """ Average pooling layer.

    Parameters
    ----------
    mode : {{'include_padding', 'exclude_padding'}}
        Gives you the choice to include or exclude padding.
        Defaults to ``include_padding``.
    {BasePooling.size}
    {BasePooling.stride_size}
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
