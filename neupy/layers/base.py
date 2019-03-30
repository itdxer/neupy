import re
import types
import inspect
from functools import partial
from collections import OrderedDict, defaultdict

import six
import numpy as np
import tensorflow as tf

from neupy.exceptions import LayerConnectionError
from neupy.core.properties import Property, TypedListProperty
from neupy.utils import as_tuple, tf_utils
from neupy.layers.graph import BaseGraph, make_one_if_possible


__all__ = ('BaseLayer', 'Identity', 'Input')


def generate_layer_name(layer):
    if not hasattr(generate_layer_name, 'counters'):
        generate_layer_name.counters = defaultdict(int)

    classname = layer.__class__.__name__
    generate_layer_name.counters[classname] += 1
    layer_id = generate_layer_name.counters[classname]

    layer_name = re.sub(r'(?<!^)(?=[A-Z][a-z_])', '-', classname)
    return "{}-{}".format(layer_name.lower(), layer_id)


class BaseLayer(BaseGraph):
    """
    Base class for the layers.

    Parameters
    ----------
    name : str or None
        Layer's name. Can be used as a reference to specific layer. When
        value specified as ``None`` than name will be generated from
        the class name. Defaults to ``None``

    Methods
    -------
    variable(value, name, shape=None, trainable=True)
        Initializes variable with specified values.

    get_output_shape(input_shape)
        Computes expected output shape from the layer based on the
        specified input shape.

    output(*inputs, **kwargs)
        Propagetes input through the layer. The ``kwargs``  variable
        might contain additional information that propages through the
        network.

    Attributes
    ----------
    variables : dict
        Variable names and their values. Dictionary can be empty in case
        if variables hasn't been created yet.
    """
    name = Property(expected_type=six.string_types)

    def __init__(self, name=None):
        # Layer by default gets intialized as a graph with single node in it
        super(BaseLayer, self).__init__(forward_graph=[(self, [])])

        if name is None:
            name = generate_layer_name(layer=self)

        self.variables = OrderedDict()
        self.name = name

        self._input_shape = tf.TensorShape(None)
        self.frozen = False

        # This decorator ensures that result produced by the
        # `output` method will be marked under layer's name scope.
        self.output = types.MethodType(
            tf_utils.class_method_name_scope(self.output), self)

    @classmethod
    def define(cls, *args, **kwargs):
        return partial(cls, *args, **kwargs)

    @property
    def input_shape(self):
        # Explicit TensorShape transformation not only ensures
        # that we have right type in the output, but also copies
        # value stored in the `_input_shape` in order to make sure
        # that no inplace update can effect original value
        return tf.TensorShape(self._input_shape)

    @input_shape.setter
    def input_shape(self, shape):
        if not self._input_shape.is_compatible_with(shape):
            raise ValueError(
                "Cannot update input shape of the layer, because it's "
                "incompatible with current input shape. Current shape: {}, "
                "New shape: {}, Layer: {}".format(
                    self._input_shape, shape, self))

        self._input_shape = tf.TensorShape(shape)

    @property
    def output_shape(self):
        return self.get_output_shape(self.input_shape)

    def get_output_shape(self, input_shape):
        return tf.TensorShape(None)

    def create_variables(self, *input_shapes):
        return NotImplemented

    def variable(self, value, name, shape=None, trainable=True):
        layer_name = 'layer/{layer_name}/{parameter_name}'.format(
            layer_name=self.name,
            parameter_name=name.replace('_', '-'))

        self.variables[name] = tf_utils.create_variable(
            value, layer_name, shape, trainable)

        return self.variables[name]

    def _repr_arguments(self, *args, **kwargs):
        def format_value(value):
            references = {
                'Variable': tf.Variable,
                'Array': np.ndarray,
                'Matrix': np.matrix,
            }

            for name, datatype in references.items():
                if isinstance(value, datatype):
                    return '<{} shape={}>'.format(name, value.shape)

            return repr(value)

        formatted_args = [str(arg) for arg in args]
        argspec = inspect.getargspec(self.__class__.__init__)

        def kwargs_priority(value):
            if value in argspec.args:
                return argspec.args.index(value)
            return float('inf')

        # Kwargs will have destroyed order of the arguments, and order in
        # the __init__ method allows to use proper order and validate names
        for name in sorted(kwargs.keys(), key=kwargs_priority):
            value = format_value(kwargs[name])
            formatted_args.append('{}={}'.format(name, value))

        return '{clsname}({formatted_args})'.format(
            clsname=self.__class__.__name__,
            formatted_args=', '.join(formatted_args))

    def __repr__(self):
        kwargs = {}

        for name in self.options:
            value = getattr(self, name)
            kwargs[name] = value

        return self._repr_arguments(**kwargs)


class Identity(BaseLayer):
    """
    Passes input through the layer without changes. Can be
    useful while defining residual networks in the network.

    Parameters
    ----------
    {BaseLayer.name}

    Methods
    -------
    {BaseLayer.Methods}

    Attributes
    ----------
    {BaseLayer.Attributes}
    """
    def get_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

    def output(self, input, **kwargs):
        return input


class Input(BaseLayer):
    """
    Layer defines network's input.

    Parameters
    ----------
    shape : int or tuple
        Shape of the input features per sample. Batch
        dimension has to be excluded from the shape.

    {BaseLayer.name}

    Methods
    -------
    {BaseLayer.Methods}

    Attributes
    ----------
    {BaseLayer.Attributes}

    Examples
    --------
    Feedforward Neural Network (FNN)

    In the example, input layer defines network that expects
    2D inputs (matrices). In other words, input to the network
    should be set of samples combined into matrix where each sample
    has 10 dimensional vector associated with it.

    >>> from neupy.layers import *
    >>> network = Input(10) >> Relu(5) >> Softmax(3)

    Convolutional Neural Network (CNN)

    In the example, input layer specified that we expect multiple
    28x28 image as an input and each image should have single
    channel (images with no color).

    >>> from neupy.layers import *
    >>> network = join(
    ...     Input((28, 28, 1)),
    ...     Convolution((3, 3, 16)) >> Relu(),
    ...     Convolution((3, 3, 16)) >> Relu(),
    ...     Reshape()
    ...     Softmax(10),
    ... )
    """
    shape = TypedListProperty(element_type=(int, type(None)))

    def __init__(self, shape, name=None):
        super(Input, self).__init__(name=name)

        if isinstance(shape, tf.TensorShape):
            shape = tf_utils.shape_to_tuple(shape)

        self.shape = as_tuple(shape)

    @BaseLayer.input_shape.getter
    def input_shape(self):
        batch_shape = tf.TensorShape([None])
        return batch_shape.concatenate(self.shape)

    def output(self, input, **kwargs):
        return input

    def get_output_shape(self, input_shape):
        if not self.input_shape.is_compatible_with(input_shape):
            raise LayerConnectionError(
                "Input layer got unexpected input shape. "
                "Received shape: {}, Expected shape: {}"
                "".format(input_shape, self.input_shape)
            )
        return self.input_shape.merge_with(input_shape)

    def __repr__(self):
        return self._repr_arguments(
            make_one_if_possible(self.shape),
            name=self.name)
