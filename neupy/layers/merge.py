import copy
from functools import reduce

import theano.tensor as T

from neupy.core.properties import IntProperty, CallableProperty
from neupy.exceptions import LayerConnectionError
from neupy.utils import as_tuple, all_equal
from .base import BaseLayer


__all__ = ('Elementwise', 'Concatenate', 'GatedAverage')


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
                "".format(self, input_shapes))

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
                        "".format(axis))

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


def exclude_index(array, index):
    """
    Copies array and exclude single element in specific
    index position.

    Parameters
    ----------
    array : list or tuple

    index : int
        Index of the value that has to be excluded from the arrray

    Returns
    -------
    list
    """
    array = list(array)
    copied_array = copy.copy(array)
    copied_array.pop(index)
    return copied_array


class GatedAverage(BaseLayer):
    """
    Using output from the gated layer weights outputs from the
    other layers and sum them.

    Parameters
    ----------
    gating_layer_index : int
        Input layers passed as a list and current variable specifies
        index in which it can find gating network. Defaults to `0`,
        which means that it expects to see gating layer in zeros position.

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
    >>>
    >>> gating_network = Input(10) > Softmax(2)
    >>> network_1 = Input(20) > Relu(10)
    >>> network_2 = Input(20) > Relu(20) > Relu(10)
    >>>
    >>> network = [gating_network, network_1, network_2] > GatedAverage()
    >>> network
    [(10,), (20,), (20,)] -> [... 8 layers ...] -> 10
    """
    gating_layer_index = IntProperty(default=0)

    def validate(self, input_shapes):
        n_input_layers = len(input_shapes)
        gating_layer_index = self.gating_layer_index

        try:
            gating_layer_shape = input_shapes[gating_layer_index]
        except IndexError:
            raise LayerConnectionError(
                "Invalid index for gating layer. Number of input "
                "layers: {}. Gating layer index: {}"
                "".format(n_input_layers, gating_layer_index))

        other_layers_shape = exclude_index(input_shapes, gating_layer_index)

        if len(gating_layer_shape) != 1:
            raise LayerConnectionError(
                "Output from the gating network should be vector. Output "
                "shape from gating layer: {!r}".format(gating_layer_shape))

        n_gating_weights = gating_layer_shape[0]
        # Note: -1 from all layers in order to exclude gating layer
        if n_gating_weights != (n_input_layers - 1):
            raise LayerConnectionError(
                "Gating layer can work only for combining only {} networks, "
                "got {} networks instead."
                "".format(n_gating_weights, (n_input_layers - 1)))

        if not all_equal(other_layers_shape):
            raise LayerConnectionError(
                "Output layer that has to be merged expect to have the "
                "same shapes. Shapes: {!r}".format(other_layers_shape))

    @property
    def output_shape(self):
        if not self.input_shape:
            return

        if self.gating_layer_index >= 0:
            # Take layer from the left side from the gating layer.
            # In case if gating layer at th zeros position then
            # it will take the last layer (-1 index).
            return self.input_shape[self.gating_layer_index - 1]

        # In case if it negative index, we take layer from the right side
        return self.input_shape[self.gating_layer_index + 1]

    def output(self, *input_values):
        gating_value = input_values[self.gating_layer_index]
        other_values = exclude_index(input_values, self.gating_layer_index)

        # Input shape is exactly the same as output shape
        n_output_dim = len(self.output_shape)

        output_values = []
        for i, other_value in enumerate(other_values):
            gate = gating_value[:, i]
            new_shape = [0] + ['x'] * n_output_dim

            output_value = T.mul(other_value, gate.dimshuffle(*new_shape))
            output_values.append(output_value)

        return sum(output_values)
