import copy
from functools import reduce

import theano.tensor as T

from neupy.core.properties import IntProperty, Property
from neupy.exceptions import LayerConnectionError
from neupy.utils import as_tuple
from .base import BaseLayer


__all__ = ('Elementwise', 'Concatenate')


class CallableProperty(Property):
    """
    Property for callable objects.

    Parameters
    ----------
    {Property.Parameters}
    """
    def validate(self, value):
        if not callable(value):
            raise ValueError("The `{}` property expected to be "
                             "callable object.".format(self.name))
        super(CallableProperty, self).validate(value)


class Elementwise(BaseLayer):
    """
    Merge multiple input layers in one with elementwise
    function.

    Parameters
    ----------
    merge_function : callable
        Callable object that accepts multiple arguments and
        combine them in one with elementwise operation.
        Defaults to ``theano.tensor.add``

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
    >>> input_1 = layers.Input(10)
    >>> input_2 = layers.Input(10)
    >>>
    >>> network = [input_1, input_2] > layers.Elementwise()
    >>>
    >>> network.input_shape
    [(10,), (10,)]
    >>> network.output_shape
    (10,)
    """
    merge_function = CallableProperty(default=T.add)

    def validate(self, input_shapes):
        n_unique_shapes = len(set(input_shapes))
        if n_unique_shapes != 1:
            raise LayerConnectionError(
                "The `{}` layer expects all input values with the "
                "same shapes. Input shapes: {}"
                "".format(self, input_shapes)
            )

    @property
    def output_shape(self):
        if self.input_shape:
            return self.input_shape[0]

    def output(self, *input_values):
        if len(input_values) == 1:
            return input_values[0]
        return reduce(self.merge_function, input_values)


class Concatenate(BaseLayer):
    """
    Concatenate multiple input layers in one based on the
    specified axes.

    Parameters
    ----------
    axis : int
        The axis along which the inputs will be joined.
        Default is ``1``.

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
    >>> input_1 = layers.Input(10)
    >>> input_2 = layers.Input(20)
    >>>
    >>> network = [input_1, input_2] > layers.Concatenate()
    >>>
    >>> network.input_shape
    [(10,), (20,)]
    >>> network.output_shape
    (30,)
    """
    axis = IntProperty(default=1)

    def validate(self, input_shapes):
        valid_shape = as_tuple(None, input_shapes[0])

        for input_shape in input_shapes[1:]:
            for axis, axis_size in enumerate(input_shape, start=1):
                if axis != self.axis and valid_shape[axis] != axis_size:
                    raise LayerConnectionError(
                        "Cannot concatenate layers. Some of them don't "
                        "match over dimension #{} (0-based indeces)."
                        "".format(axis)
                    )

    @property
    def output_shape(self):
        if not self.input_shape:
            return

        axis = self.axis - 1  # because we do not include #0 dim
        input_shapes = copy.copy(self.input_shape)
        output_shape = list(input_shapes.pop(0))

        for input_shape in input_shapes:
            output_shape[axis] += input_shape[axis]

        return tuple(output_shape)

    def output(self, *input_values):
        return T.concatenate(input_values, axis=self.axis)
