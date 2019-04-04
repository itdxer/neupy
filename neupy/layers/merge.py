import copy
from functools import reduce

import tensorflow as tf

from neupy.core.properties import FunctionWithOptionsProperty, IntProperty
from neupy.exceptions import LayerConnectionError
from neupy.utils import tf_utils
from .base import BaseLayer


__all__ = ('Elementwise', 'Concatenate', 'GatedAverage')


class Elementwise(BaseLayer):
    """
    Layer merges multiple input with elementwise function and generate
    single output. Each input to this layer should have exactly the same
    shape, otherwise it won't be possible to apply elementwise operation.

    Parameters
    ----------
    merge_function : callable or {{``add``, ``multiply``}}
        Callable object that accepts two inputs and
        combines them in value using elementwise operation.

        - ``add`` - Sum all the inputs. Alias to ``tf.add``.

        - ``multiply`` - Multiplies all the inputs. Alias to ``tf.multiply``.

        - Custom function requires to have two input arguments.

        .. code-block:: python

            def subtraction(x, y):
                return x - y

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
    [(?, 10), (?, 10)] -> [... 3 layers ...] -> (?, 10)
    """
    merge_function = FunctionWithOptionsProperty(choices={
        'add': tf.add,
        'multiply': tf.multiply,
    })

    def __init__(self, merge_function='add', name=None):
        super(Elementwise, self).__init__(name=name)
        self.original_function = merge_function
        self.merge_function = merge_function

    def get_output_shape(self, *input_shapes):
        input_shapes = [tf.TensorShape(shape) for shape in input_shapes]
        first_shape = input_shapes[0]

        if len(input_shapes) < 2:
            raise LayerConnectionError(
                "Layer `{}` expected multiple inputs. Input shapes: {}"
                "".format(self.name, tf_utils.shape_to_tuple(input_shapes)))

        if any(shape.ndims is None for shape in input_shapes):
            return tf.TensorShape(None)

        for shape in input_shapes:
            if not shape.is_compatible_with(first_shape):
                formatted_shapes = tf_utils.shape_to_tuple(input_shapes)
                raise LayerConnectionError(
                    "Input shapes to the `{}` layer have incompatible shapes. "
                    "Input shapes: {}, Layer: {}"
                    "".format(self.name, formatted_shapes, self))

        return first_shape

    def output(self, *inputs, **kwargs):
        return reduce(self.merge_function, inputs)

    def __repr__(self):
        return self._repr_arguments(
            repr(self.original_function), name=self.name)


class Concatenate(BaseLayer):
    """
    Concatenate multiple inputs into one. Inputs will be concatenated over
    the specified axis (controlled with parameter ``axis``).

    Parameters
    ----------
    axis : int
        The axis along which the inputs will be concatenated.
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
    >>> network = (Input(10) | Input(20)) >> Concatenate()
    [(?, 10), (?, 20)] -> [... 3 layers ...] -> (?, 30)
    """
    axis = IntProperty()

    def __init__(self, axis=-1, name=None):
        super(Concatenate, self).__init__(name=name)
        self.axis = axis

    def get_output_shape(self, *input_shapes):
        input_shapes = [tf.TensorShape(shape) for shape in input_shapes]
        # The axis value has 0-based indices where 0s index points
        # to the batch dimension of the input. Shapes in the neupy
        # do not store information about the batch and we need to
        # put None value on the 0s position.
        valid_shape = input_shapes[0]

        if any(shape.ndims is None for shape in input_shapes):
            return tf.TensorShape(None)

        # Avoid using negative indices
        possible_axes = list(range(len(valid_shape)))
        concat_axis = possible_axes[self.axis]

        for input_shape in input_shapes[1:]:
            if len(valid_shape) != len(input_shape):
                raise LayerConnectionError(
                    "Cannot concatenate layers, because inputs have "
                    "different number of dimensions. Shapes: {} and {}"
                    "".format(valid_shape, input_shape))

            for axis, axis_size in enumerate(input_shape):
                if axis != concat_axis and valid_shape[axis] != axis_size:
                    raise LayerConnectionError(
                        "Cannot concatenate layers, because some of them "
                        "don't match over dimension #{} (0-based indices). "
                        "Shapes: {} and {}"
                        "".format(axis, valid_shape, input_shape))

        output_shape = input_shapes.pop(0)
        output_shape = [dim.value for dim in output_shape.dims]

        for input_shape in input_shapes:
            output_shape[self.axis] += input_shape[self.axis]

        return tf.TensorShape(output_shape)

    def output(self, *inputs, **kwargs):
        return tf.concat(inputs, axis=self.axis)


def exclude_index(array, index):
    """
    Copies array and exclude single element in specific
    index position.

    Parameters
    ----------
    array : list or tuple

    index : int
        Index of the value that has to be excluded from the array

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
    Layer uses applies weighted elementwise addition to multiple outputs.
    Weight can be control using separate input known as **gate**. Number
    of outputs from the gate has to be equal to the number of networks,
    since each value from the weight will be a weight per each network.

    Layer expects gate as a first input, but it can be controlled with
    the ``gate_index`` parameter.

    Parameters
    ----------
    gate_index : int
        Input layers passed as a list and current variable specifies
        index in which it can find gating network. Defaults to ``0``,
        which means that it expects to see gating layer in first position.

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
    >>> gate = Input(10) >> Softmax(2)
    >>> net1 = Input(20) >> Relu(10)
    >>> net2 = Input(20) >> Relu(20) >> Relu(10)
    >>>
    >>> network = (gate | net1 | net2) >> GatedAverage()
    >>> network
    [(10,), (20,), (20,)] -> [... 8 layers ...] -> 10
    """
    gate_index = IntProperty(default=0)

    def __init__(self, gate_index=0, name=None):
        super(GatedAverage, self).__init__(name=name)
        self.gate_index = gate_index

    def fail_if_shape_invalid(self, input_shapes):
        n_input_layers = len(input_shapes)

        try:
            gate_shape = input_shapes[self.gate_index]
        except IndexError:
            raise LayerConnectionError(
                "Invalid index for gating layer. Number of input "
                "layers: {}. Gating layer index: {}"
                "".format(n_input_layers, self.gate_index))

        other_shapes = exclude_index(input_shapes, self.gate_index)
        if gate_shape and len(gate_shape) != 2:
            raise LayerConnectionError(
                "Output from the gating network should be 2-dimensional. "
                "Output shape from gating layer: {!r}"
                "".format(gate_shape))

        n_expected_networks = gate_shape[-1]
        # Note: -1 from all layers in order to exclude gating layer
        if n_expected_networks != (n_input_layers - 1):
            raise LayerConnectionError(
                "Gating layer can work only for combining only {} networks, "
                "got {} networks instead."
                "".format(n_expected_networks, (n_input_layers - 1)))

        for shape in other_shapes:
            if not shape.is_compatible_with(other_shapes[0]):
                raise LayerConnectionError(
                    "Output layer that has to be merged expect to "
                    "have the same shapes. Shapes: {!r}"
                    "".format(tf_utils.shape_to_tuple(other_shapes)))

    def get_output_shape(self, *input_shapes):
        input_shapes = [tf.TensorShape(shape) for shape in input_shapes]

        if any(shape.ndims is None for shape in input_shapes):
            return tf.TensorShape(None)

        self.fail_if_shape_invalid(input_shapes)

        if self.gate_index >= 0:
            # Take layer from the left side from the gating layer.
            # In case if gating layer at the zeros position then
            # it will take the last layer (-1 index).
            return input_shapes[self.gate_index - 1]

        # In case if it negative index, we take layer from the right side
        return input_shapes[self.gate_index + 1]

    def output(self, *inputs, **kwargs):
        gating_value = inputs[self.gate_index]
        other_values = exclude_index(inputs, self.gate_index)
        output_values = []

        for i, other_value in enumerate(other_values):
            n_feature_dim = other_value.shape.ndims - 1
            gate = tf.reshape(gating_value[:, i], [-1] + [1] * n_feature_dim)
            output_value = tf.multiply(other_value, gate)
            output_values.append(output_value)

        return sum(output_values)
