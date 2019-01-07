import copy
from functools import reduce

import tensorflow as tf

from neupy.core.properties import FunctionWithOptionsProperty, IntProperty
from neupy.exceptions import LayerConnectionError
from neupy.utils import as_tuple, tf_utils
from .base import BaseLayer


__all__ = ('Elementwise', 'Concatenate', 'GatedAverage')


class Elementwise(BaseLayer):
    """
    Merge multiple input layers elementwise function and generate
    single output. Each input to this layer should have exactly the
    same shape.

    Parameters
    ----------
    merge_function : callable or {{``add``, ``mul``}}
        Callable object that accepts multiple arguments and
        combine them in one with elementwise operation.
        Defaults to ``add``.

    {BaseLayer.name}

    Methods
    -------
    {BaseLayer.Methods}

    Attributes
    ----------
    {BaseLayer.Attributes}

    Examples
    --------
    >>> from neupy import layers
    >>> network = (Input(10) | Input(10)) >> Elementwise('add')
    >>>
    >>> network.input_shape
    [(10,), (10,)]
    >>> network.output_shape
    (10,)
    """
    merge_function = FunctionWithOptionsProperty(choices={
        'add': tf.add,
        'multiply': tf.multiply,
    })

    def __init__(self, merge_function='add', name=None):
        super(Elementwise, self).__init__(name=name)
        self.merge_function = merge_function

    def get_output_shape(self, *input_shapes):
        input_shapes = [tf.TensorShape(shape) for shape in input_shapes]
        first_shape = input_shapes[0]

        if any(shape != first_shape for shape in input_shapes):
            raise LayerConnectionError(
                "The `{}` layer expects all input values with "
                "exactly the same shapes. Input shapes: {}"
                "".format(self, input_shapes))

        return first_shape

    def output(self, *inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 1:
            raise LayerConnectionError(
                "Layer `{}` expected multiple inputs".format(self.name))

        return reduce(self.merge_function, inputs)


class Concatenate(BaseLayer):
    """
    Concatenate multiple input layers in one based on the
    specified axes.

    Parameters
    ----------
    axis : int
        The axis along which the inputs will be joined.
        Default is ``-1``.

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
    >>>
    >>> network = (Input(10) | Input(20)) >> Concatenate()
    >>> network.input_shape
    [(10,), (20,)]
    >>> network.output_shape
    (30,)
    """
    axis = IntProperty()

    def __init__(self, axis=-1, name=None):
        super(Concatenate, self).__init__(name=name)
        self.axis = axis

    def get_output_shape(self, *input_shapes):
        input_shapes = [tf.TensorShape(shape) for shape in input_shapes]
        # The axis value has 0-based indeces where 0s index points
        # to the batch dimension of the input. Shapes in the neupy
        # do not store information about the batch and we need to
        # put None value on the 0s position.
        valid_shape = tf_utils.add_batch_dim(input_shapes[0])

        # Avoid using negative indeces
        possible_axes = list(range(len(valid_shape)))
        concat_axis = possible_axes[self.axis]

        for input_shape in input_shapes[1:]:
            if len(input_shapes[0]) != len(input_shape):
                raise LayerConnectionError(
                    "Cannot concatenate layers, because inputs have "
                    "different number of dimensions. Shapes: {} and {}"
                    "".format(input_shapes[0], input_shape))

            for axis, axis_size in enumerate(input_shape, start=1):
                if axis != concat_axis and valid_shape[axis] != axis_size:
                    raise LayerConnectionError(
                        "Cannot concatenate layers, because some of them "
                        "don't match over dimension #{} (0-based indeces)."
                        "Shapes: {} and {}"
                        "".format(axis, input_shapes[0], input_shape))

        axis = self.axis
        input_shapes = list(input_shapes)
        output_shape = list(input_shapes.pop(0))

        for input_shape in input_shapes:
            output_shape[axis] += input_shape[axis]

        return tf.TensorShape(output_shape)

    def output(self, *inputs):
        return tf.concat(inputs, axis=self.axis)


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
    >>> gate = Input(10) > Softmax(2)
    >>> net1 = Input(20) > Relu(10)
    >>> net2 = Input(20) > Relu(20) > Relu(10)
    >>>
    >>> network = (gate | net1 | net2) >> GatedAverage()
    >>> network
    [(10,), (20,), (20,)] -> [... 8 layers ...] -> 10
    """
    gating_layer_index = IntProperty(default=0)

    def __init__(self, gating_layer_index=0, name=None):
        super(GatedAverage, self).__init__(name=name)
        self.gating_layer_index = gating_layer_index

    def fail_if_shape_invalid(self, input_shapes):
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

        if gating_layer_shape and len(gating_layer_shape) != 1:
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

        if any(shape != other_layers_shape[0] for shape in other_layers_shape):
            raise LayerConnectionError(
                "Output layer that has to be merged expect to have the "
                "same shapes. Shapes: {!r}".format(other_layers_shape))

    def get_output_shape(self, *input_shapes):
        input_shapes = [tf.TensorShape(shape) for shape in input_shapes]
        self.fail_if_shape_invalid(input_shapes)

        if self.gating_layer_index >= 0:
            # Take layer from the left side from the gating layer.
            # In case if gating layer at th zeros position then
            # it will take the last layer (-1 index).
            return input_shapes[self.gating_layer_index - 1]

        # In case if it negative index, we take layer from the right side
        return input_shapes[self.gating_layer_index + 1]

    def output(self, input_values):
        gating_value = input_values[self.gating_layer_index]
        other_values = exclude_index(input_values, self.gating_layer_index)

        # Input shape is exactly the same as output shape
        n_output_dim = len(self.output_shape)
        output_values = []

        for i, other_value in enumerate(other_values):
            output_value = tf.multiply(
                other_value,
                tf.reshape(
                    gating_value[:, i],
                    [-1] + [1] * n_output_dim
                ),
            )
            output_values.append(output_value)
        return sum(output_values)
