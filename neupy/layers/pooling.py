import theano.tensor as T
from theano.tensor.signal import pool

from neupy.utils import as_tuple
from neupy.core.properties import TypedListProperty, ChoiceProperty, Property
from neupy.exceptions import LayerConnectionError
from .base import BaseLayer
from .convolutions import StrideProperty


__all__ = ('MaxPooling', 'AveragePooling', 'Upscale', 'GlobalPooling')


class PaddingProperty(TypedListProperty):
    expected_type = as_tuple(TypedListProperty.expected_type, int)

    def __set__(self, instance, value):
        if isinstance(value, int):
            value = (value, value)
        super(PaddingProperty, self).__set__(instance, value)


def pooling_output_shape(dimension_size, pool_size, padding, stride,
                         ignore_border=True):
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

    ignore_border : bool
        Defaults to ``True``.

    Returns
    -------
    int
    """
    if dimension_size is None:
        return None

    if ignore_border:
        output_size = dimension_size + 2 * padding - pool_size + 1
        output_size = (output_size + stride - 1) // stride

    elif stride >= pool_size:
        output_size = (dimension_size + stride - 1) // stride

    else:
        output_size = (dimension_size - pool_size + stride - 1) // stride
        output_size = max(1, output_size + 1)

    return output_size


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

    padding : tuple or int
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
    padding = PaddingProperty(default=0, element_type=int, n_elements=2)
    ignore_border = Property(default=True, expected_type=bool)

    def __init__(self, size, **options):
        super(BasePooling, self).__init__(size=size, **options)

        if not self.ignore_border and self.padding != (0, 0):
            raise ValueError("Cannot set padding parameter equal to {} while "
                             "``ignore_border`` is equal to ``False``"
                             "".format(self.padding))

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
        row_padding, col_padding = self.padding

        output_rows = pooling_output_shape(rows, row_filter_size, row_padding,
                                           row_stride, self.ignore_border)
        output_cols = pooling_output_shape(cols, col_filter_size, col_padding,
                                           col_stride, self.ignore_border)

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
    def output(self, input_value):
        return pool.pool_2d(input_value, ds=self.size, mode='max',
                            ignore_border=self.ignore_border,
                            st=self.stride, padding=self.padding)


class AveragePooling(BasePooling):
    """
    Average pooling layer.

    Parameters
    ----------
    mode : {{``include_padding``, ``exclude_padding``}}
        Give a choice to include or exclude padding.
        Defaults to ``include_padding``.

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
    mode = ChoiceProperty(
        default='include_padding',
        choices={
            'include_padding': 'average_inc_pad',
            'exclude_padding': 'average_exc_pad'
        }
    )

    def output(self, input_value):
        return pool.pool_2d(input_value, ds=self.size, mode=self.mode,
                            ignore_border=self.ignore_border,
                            st=self.stride, padding=self.padding)


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
                "3 feature dimensions (channel, height, width)"
            )

    @property
    def output_shape(self):
        if self.input_shape is None:
            return None

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
    function = Property(default=T.mean)

    @property
    def output_shape(self):
        if self.input_shape is not None:
            return as_tuple(self.input_shape[0])

    def output(self, input_value):
        if input_value.ndim in (1, 2):
            return input_value

        agg_axis = range(2, input_value.ndim)
        return self.function(input_value, axis=list(agg_axis))
