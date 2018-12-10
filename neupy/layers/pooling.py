from __future__ import division

import math

import tensorflow as tf

from neupy.utils import as_tuple, tf_repeat
from neupy.core.properties import (TypedListProperty, ChoiceProperty,
                                   FunctionWithOptionsProperty)
from neupy.exceptions import LayerConnectionError
from .base import BaseLayer
from .convolutions import Spatial2DProperty


__all__ = ('MaxPooling', 'AveragePooling', 'Upscale', 'GlobalPooling')


def pooling_output_shape(dimension_size, pool_size, padding, stride):
    """
    Computes output shape for pooling operation.

    Parameters
    ----------
    dimension_size : int
        Size of the dimension. Typically it's image's
        weight or height.

    filter_size : int
        Size of the pooling filter.

    padding : int
        Size of the zero-padding.

    stride : int
        Stride size.

    Returns
    -------
    int
    """
    if dimension_size is None:
        return None

    if padding in ('SAME', 'same'):
        return int(math.ceil(dimension_size / stride))

    elif padding in ('VALID', 'valid'):
        return int(math.ceil((dimension_size - pool_size + 1) / stride))

    raise ValueError(
        "{!r} is unknown convolution's padding value".format(padding))


class BasePooling(BaseLayer):
    """
    Base class for the pooling layers.

    Parameters
    ----------
    size : tuple with 2 integers
        Factor by which to downscale ``(vertical, horizontal)``.
        ``(2, 2)`` will halve the image in each dimension.

    stride : tuple or int.
        Stride size, which is the number of shifts over
        rows/cols to get the next pool region. If stride is
        None, it is considered equal to ds (no overlap on
        pooling regions).

    padding : {{``valid``, ``same``}}
        ``(pad_h, pad_w)``, pad zeros to extend beyond four borders of
        the images, pad_h is the size of the top and bottom margins,
        and pad_w is the size of the left and right margins.

    {BaseLayer.Parameters}

    Methods
    -------
    {BaseLayer.Methods}

    Attributes
    ----------
    {BaseLayer.Attributes}
    """
    size = TypedListProperty(required=True, element_type=int)
    stride = Spatial2DProperty(default=None)
    padding = ChoiceProperty(default='VALID', choices=(
        'SAME', 'VALID', 'same', 'valid'))

    pooling_type = None

    def __init__(self, size, **options):
        super(BasePooling, self).__init__(size=size, **options)

    def validate(self, input_shape):
        if len(input_shape) != 3:
            raise LayerConnectionError(
                "Pooling layer expects an input with 3 "
                "dimensions, got {} with shape {}"
                "".format(len(input_shape), input_shape)
            )

    @property
    def output_shape(self):
        if self.input_shape is None:
            return None

        rows, cols, n_kernels = self.input_shape
        row_filter_size, col_filter_size = self.size

        stride = self.size if self.stride is None else self.stride
        row_stride, col_stride = stride

        output_rows = pooling_output_shape(
            rows, row_filter_size, self.padding, row_stride)

        output_cols = pooling_output_shape(
            cols, col_filter_size, self.padding, col_stride)

        # In python 2, we can get float number after rounding procedure
        # and it might break processing in the subsequent layers.
        return (output_rows, output_cols, n_kernels)

    def output(self, input_value):
        return tf.nn.pool(
            input_value,
            self.size,
            pooling_type=self.pooling_type,
            padding=self.padding.upper(),
            strides=self.stride or self.size,
            data_format="NHWC",
        )

    def __repr__(self):
        return '{name}({size})'.format(
            name=self.__class__.__name__,
            size=self.size,
        )


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

    Examples
    --------
    2D pooling

    >>> from neupy import layers
    >>>
    >>> network = layers.join(
    ...     layers.Input((10, 10, 3)),
    ...     layers.MaxPooling((2, 2)),
    ... )
    >>> network.output_shape
    (3, 5, 5)

    1D pooling

    >>> from neupy import layers
    >>>
    >>> network = layers.join(
    ...     layers.Input((30, 10)),
    ...     layers.Reshape((10, 1, 30)),
    ...     layers.MaxPooling((2, 1)),
    ... )
    >>> network.output_shape
    (10, 15, 1)
    """
    pooling_type = 'MAX'


class AveragePooling(BasePooling):
    """
    Average pooling layer.

    Parameters
    ----------
    {BasePooling.Parameters}

    Methods
    -------
    {BasePooling.Methods}

    Attributes
    ----------
    {BasePooling.Attributes}

    Examples
    --------
    2D pooling

    >>> from neupy import layers
    >>>
    >>> network = layers.join(
    ...     layers.Input((10, 10, 3)),
    ...     layers.AveragePooling((2, 2)),
    ... )
    >>> network.output_shape
    (3, 5, 5)

    1D pooling

    >>> from neupy import layers
    >>>
    >>> network = layers.join(
    ...     layers.Input((30, 10)),
    ...     layers.Reshape((10, 1, 30)),
    ...     layers.AveragePooling((2, 1)),
    ... )
    >>> network.output_shape
    (10, 15, 1)
    """
    pooling_type = 'AVG'


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

    {BaseLayer.Parameters}

    Methods
    -------
    {BaseLayer.Methods}

    Attributes
    ----------
    {BaseLayer.Attributes}

    Examples
    --------
    >>> from neupy.layers import *
    >>> network = Input((10, 10, 3)) > Upscale((2, 2))
    >>> network.output_shape
    (3, 20, 20)
    """
    scale = ScaleFactorProperty(required=True, n_elements=2)

    def __init__(self, scale, **options):
        super(Upscale, self).__init__(scale=scale, **options)

    def validate(self, input_shape):
        if len(input_shape) != 3:
            raise LayerConnectionError(
                "Upscale layer should have an input value with "
                "3 feature dimensions (channel, height, width)")

    @property
    def output_shape(self):
        if self.input_shape is None:
            return None

        height, width, channel = self.input_shape
        height_scale, width_scale = self.scale

        return (height_scale * height, width_scale * width, channel)

    def output(self, input_value):
        return tf_repeat(input_value, as_tuple(1, self.scale, 1))


class GlobalPooling(BaseLayer):
    """
    Global pooling layer.

    Parameters
    ----------
    function : {{``avg``, ``max``}} or callable
        Common functions has been predefined for the user.
        These options are available:

        - ``avg`` - For average global pooling. The same as
          ``tf.reduce_mean``.

        - ``max`` - For average global pooling. The same as
          ``tf.reduce_max``.

        Parameters also excepts custom functions that have
        following format.

        .. code-block:: python

            def agg_func(x, axis=None):
                pass

        Defaults to ``avg``.

    {BaseLayer.Parameters}

    Methods
    -------
    {BaseLayer.Methods}

    Attributes
    ----------
    {BaseLayer.Attributes}

    Examples
    --------
    >>> from neupy.layers import *
    >>> network = Input((4, 4, 16)) > GlobalPooling('avg')
    >>> network.output_shape
    (16,)
    """
    function = FunctionWithOptionsProperty(choices={
        'avg': tf.reduce_mean,
        'max': tf.reduce_max,
    })

    def __init__(self, function, *args, **kwargs):
        super(GlobalPooling, self).__init__(
            *args, **dict(kwargs, function=function))

    @property
    def output_shape(self):
        if self.input_shape is not None:
            return as_tuple(self.input_shape[-1])

    def output(self, input_value):
        ndims = len(input_value.shape)

        if ndims in (1, 2):
            return input_value

        # All dimensions except first and last
        agg_axis = range(1, ndims - 1)
        return self.function(input_value, axis=list(agg_axis))
