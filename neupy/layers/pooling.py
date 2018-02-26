import math

import tensorflow as tf

from neupy.utils import as_tuple
from neupy.core.properties import TypedListProperty, ChoiceProperty, Property
from neupy.exceptions import LayerConnectionError
from .base import BaseLayer
from .convolutions import StrideProperty


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

    if padding == 'SAME':
        return math.ceil(dimension_size / stride)

    elif padding == 'VALID':
        return math.ceil((dimension_size - pool_size + 1) / stride)

    raise ValueError("`{!r}` is unknown convolution's padding value"
                     "".format(padding))


class BasePooling(BaseLayer):
    """
    Base class for the pooling layers.

    Parameters
    ----------
    size : tuple with 2 integers
        Factor by which to downscale (vertical, horizontal).
        (2, 2) will halve the image in each dimension.

    stride : tuple or int.
        Stride size, which is the number of shifts over
        rows/cols to get the next pool region. If stride is
        None, it is considered equal to ds (no overlap on
        pooling regions).

    padding : {{`VALID`, `SAME`}}
        (pad_h, pad_w), pad zeros to extend beyond four borders of
        the images, pad_h is the size of the top and bottom margins,
        and pad_w is the size of the left and right margins.

    ignore_border : bool
        When ``True``, ``(5, 5)`` input with size ``(2, 2)``
        will generate a `(2, 2)` output. ``(3, 3)`` otherwise.
        Defaults to ``True``.

    {BaseLayer.Parameters}

    Methods
    -------
    {BaseLayer.Methods}

    Attributes
    ----------
    {BaseLayer.Attributes}
    """
    size = TypedListProperty(required=True, element_type=int)
    stride = StrideProperty(default=None)
    padding = ChoiceProperty(default='VALID', choices=('SAME', 'VALID'))
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

        n_kernels, rows, cols = self.input_shape
        row_filter_size, col_filter_size = self.size

        stride = self.size if self.stride is None else self.stride
        row_stride, col_stride = stride

        output_rows = pooling_output_shape(
            rows, row_filter_size, self.padding, row_stride)

        output_cols = pooling_output_shape(
            cols, col_filter_size, self.padding, col_stride)

        return (n_kernels, output_rows, output_cols)

    def output(self, input_value):
        # TODO: transpose added only for convenient transition between
        # tensroflow and theatno. I will remove it later.
        input_value = tf.transpose(input_value, (0, 2, 3, 1))
        output = tf.nn.pool(
            input_value,
            self.size,
            pooling_type=self.pooling_type,
            padding=self.padding,
            strides=self.stride or self.size,
        )
        return tf.transpose(output, (0, 3, 1, 2))

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
    ...     layers.Input((3, 10, 10)),
    ...     layers.MaxPooling((2, 2)),
    ... )
    >>> network.output_shape
    (3, 5, 5)

    1D pooling

    >>> from neupy import layers
    >>>
    >>> network = layers.join(
    ...     layers.Input((10, 30)),
    ...     layers.Reshape((10, 30, 1)),
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
    ...     layers.Input((3, 10, 10)),
    ...     layers.AveragePooling((2, 2)),
    ... )
    >>> network.output_shape
    (3, 5, 5)

    1D pooling

    >>> from neupy import layers
    >>>
    >>> network = layers.join(
    ...     layers.Input((10, 30)),
    ...     layers.Reshape((10, 30, 1)),
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


def tf_repeat(tensor, repeats):
    """
    Repeat elements of an tensor. The same as ``numpy.repeat``.

    Parameters
    ----------
    input : tensor
    repeats: list, tuple
        Number of repeat for each dimension, length must be the
        same as the number of dimensions in input.

    Returns
    -------
    tensor
        Has the same type as input. Has the shape
        of ``tensor.shape * repeats``.
    """
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = as_tuple(1, repeats)
        tiled_tensor = tf.tile(expanded_tensor, multiples)
        repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tesnor


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
    >>> from neupy import layers
    >>>
    >>> network = layers.join(
    ...     layers.Input((3, 10, 10)),
    ...     layers.Upscale((2, 2)),
    ... )
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

        channel, height, width = self.input_shape
        height_scale, width_scale = self.scale

        return (channel, height_scale * height, width_scale * width)

    def output(self, input_value):
        if all(value == 1 for value in self.scale):
            return input_value
        return tf_repeat(input_value, as_tuple(1, 1, self.scale))


class GlobalPooling(BaseLayer):
    """
    Global pooling layer.

    Parameters
    ----------
    function : callable
        Function that aggregates over dimensions.
        Defaults to ``theano.tensor.mean``.

        .. code-block:: python

            def agg_func(x, axis=None):
                pass

    {BaseLayer.Parameters}

    Methods
    -------
    {BaseLayer.Methods}

    Attributes
    ----------
    {BaseLayer.Attributes}

    Examples
    --------
    >>> from neupy import layers
    >>>
    >>> network = layers.join(
    ...     layers.Input((16, 4, 4)),
    ...     layers.GlobalPooling(),
    ... )
    >>> network.output_shape
    (16,)
    """
    function = Property(default=tf.reduce_mean)

    @property
    def output_shape(self):
        if self.input_shape is not None:
            return as_tuple(self.input_shape[0])

    def output(self, input_value):
        if input_value.ndim in (1, 2):
            return input_value

        agg_axis = range(2, input_value.ndim)
        return self.function(input_value, axis=list(agg_axis))
