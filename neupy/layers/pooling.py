from __future__ import division

import math

import tensorflow as tf

from neupy.utils import as_tuple, tf_utils
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
    if isinstance(dimension_size, tf.Dimension):
        dimension_size = dimension_size.value

    if dimension_size is None:
        return None

    if padding in ('SAME', 'same'):
        return int(math.ceil(dimension_size / stride))

    elif padding in ('VALID', 'valid'):
        return int(math.ceil((dimension_size - pool_size + 1) / stride))

    raise ValueError(
        "{!r} is unknown padding value for pooling".format(padding))


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
        ``None``, it is considered equal to ``size`` (no overlap on
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
    stride = Spatial2DProperty(allow_none=True)
    padding = ChoiceProperty(choices=('SAME', 'VALID', 'same', 'valid'))
    pooling_type = None

    def __init__(self, size, stride=None, padding='valid', name=None):
        super(BasePooling, self).__init__(name=name)

        self.size = size
        self.stride = stride
        self.padding = padding

    def fail_if_shape_invalid(self, input_shape):
        if input_shape and input_shape.ndims != 4:
            raise LayerConnectionError(
                "Pooling layer expects an input with 4 "
                "dimensions, got {} with shape {}. Layer: {}"
                "".format(len(input_shape), input_shape, self))

    def get_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)

        if input_shape.ndims is None:
            return tf.TensorShape((None, None, None, None))

        self.fail_if_shape_invalid(input_shape)

        n_samples, rows, cols, n_kernels = input_shape
        row_filter_size, col_filter_size = self.size

        stride = self.size if self.stride is None else self.stride
        row_stride, col_stride = stride

        output_rows = pooling_output_shape(
            rows, row_filter_size, self.padding, row_stride)

        output_cols = pooling_output_shape(
            cols, col_filter_size, self.padding, col_stride)

        # In python 2, we can get float number after rounding procedure
        # and it might break processing in the subsequent layers.
        return tf.TensorShape((n_samples, output_rows, output_cols, n_kernels))

    def output(self, input_value, **kwargs):
        return tf.nn.pool(
            input_value,
            self.size,
            pooling_type=self.pooling_type,
            padding=self.padding.upper(),
            strides=self.stride or self.size,
            data_format="NHWC")

    def __repr__(self):
        return self._repr_arguments(
            self.size,
            name=self.name,
            stride=self.stride,
            padding=self.padding,
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

    >>> from neupy.layers import *
    >>> network = join(
    ...     Input((10, 10, 3)),
    ...     MaxPooling((2, 2)),
    ... )
    >>> network
    (?, 10, 10, 3) -> [... 2 layers ...] -> (?, 5, 5, 3)

    1D pooling

    >>> from neupy.layers import *
    >>> network = join(
    ...     Input((30, 10)),
    ...     Reshape((30, 1, 10)),
    ...     MaxPooling((2, 1)),
    ...     Reshape((-1, 10))
    ... )
    >>> network
    (?, 30, 10) -> [... 4 layers ...] -> (?, 15, 10)
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

    >>> from neupy.layers import *
    >>> network = join(
    ...     Input((10, 10, 3)),
    ...     AveragePooling((2, 2)),
    ... )
    >>> network
    (?, 10, 10, 3) -> [... 2 layers ...] -> (?, 5, 5, 3)

    1D pooling

    >>> from neupy.layers import *
    >>> network = join(
    ...     Input((30, 10)),
    ...     Reshape((30, 1, 10)),
    ...     AveragePooling((2, 1)),
    ...     Reshape((-1, 10))
    ... )
    >>> network
    (?, 30, 10) -> [... 4 layers ...] -> (?, 15, 10)
    """
    pooling_type = 'AVG'


class Upscale(BaseLayer):
    """
    Upscales input over two axis (height and width).

    Parameters
    ----------
    scale : int or tuple with two int
        Scaling factor for the input value. In the tuple first
        parameter identifies scale of the height and the second
        one of the width.

    {BaseLayer.name}

    Methods
    -------
    {BaseLayer.Methods}

    Attributes
    ----------
    {BaseLayer.Attributes}

    Examples
    --------
    >>> from neupy.layers import *
    >>> network = Input((10, 10, 3)) >> Upscale((2, 2))
    (?, 10, 10, 3) -> [... 2 layers ...] -> (?, 20, 20, 3)
    """
    scale = TypedListProperty(n_elements=2)

    def __init__(self, scale, name=None):
        super(Upscale, self).__init__(name=name)

        if isinstance(scale, int):
            scale = as_tuple(scale, scale)

        if any(element <= 0 for element in scale):
            raise ValueError(
                "Only positive integers are allowed for scale")

        self.scale = scale

    def fail_if_shape_invalid(self, input_shape):
        if input_shape and input_shape.ndims != 4:
            raise LayerConnectionError(
                "Upscale layer should have an input value with 4 dimensions "
                "(batch, height, width, channel), got input with {} "
                "dimensions instead. Shape: {}"
                "".format(input_shape.ndims, input_shape))

    def get_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self.fail_if_shape_invalid(input_shape)

        if input_shape.ndims is None:
            return tf.TensorShape((None, None, None, None))

        n_samples, height, width, channel = input_shape
        height_scale, width_scale = self.scale

        return tf.TensorShape([
            n_samples,
            height_scale * height,
            width_scale * width,
            channel,
        ])

    def output(self, input_value, **kwargs):
        input_value = tf.convert_to_tensor(input_value, dtype=tf.float32)
        self.fail_if_shape_invalid(input_value.shape)
        return tf_utils.repeat(input_value, as_tuple(1, self.scale, 1))

    def __repr__(self):
        return self._repr_arguments(self.scale, name=self.name)


class GlobalPooling(BaseLayer):
    """
    Global pooling layer.

    Parameters
    ----------
    function : {{``avg``, ``max``, ``sum``}} or callable
        Common functions has been predefined for the user.
        These options are available:

        - ``avg`` - For average global pooling. The same as
          ``tf.reduce_mean``.

        - ``max`` - For max global pooling. The same as
          ``tf.reduce_max``.

        - ``sum`` - For sum global pooling. The same as
          ``tf.reduce_sum``.

        Parameter also excepts custom functions that have
        following format.

        .. code-block:: python

            def agg_func(x, axis=None):
                pass

        Defaults to ``avg``.

    {BaseLayer.name}

    Methods
    -------
    {BaseLayer.Methods}

    Attributes
    ----------
    {BaseLayer.Attributes}

    Examples
    --------
    >>> from neupy.layers import *
    >>> network = Input((4, 4, 16)) >> GlobalPooling('avg')
    (?, 4, 4, 16) -> [... 2 layers ...] -> (?, 16)
    """
    function = FunctionWithOptionsProperty(choices={
        'avg': tf.reduce_mean,
        'max': tf.reduce_max,
        'sum': tf.reduce_sum,
    })

    def __init__(self, function, name=None):
        super(GlobalPooling, self).__init__(name=name)
        self.original_function = function
        self.function = function

    def get_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[-1]])

    def output(self, input_value, **kwargs):
        input_value = tf.convert_to_tensor(input_value, dtype=tf.float32)
        ndims = len(input_value.shape)

        if ndims == 2:
            return input_value

        # All dimensions except first and last
        agg_axis = range(1, ndims - 1)
        return self.function(input_value, axis=list(agg_axis))

    def __repr__(self):
        return self._repr_arguments(
            repr(self.original_function), name=self.name)
