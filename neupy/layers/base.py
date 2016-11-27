import re

import six
import theano
import theano.tensor as T

from neupy import init
from neupy.utils import asfloat, as_tuple
from neupy.core.config import Configurable
from neupy.core.properties import (TypedListProperty, IntProperty, Property,
                                   ParameterProperty)
from neupy.layers.connections import BaseConnection


__all__ = ('BaseLayer', 'ParameterBasedLayer', 'Input', 'ResidualConnection')


def next_identifier(identifiers):
    """
    Find next identifier.

    Parameters
    ----------
    identifiers : list of int
        List of identifiers

    Returns
    -------
    int
        Next identifier that doesn't appears in
        the list.
    """
    if not identifiers:
        return 1

    max_identifier = max(identifiers)
    return max_identifier + 1


def generate_layer_name(layer):
    """ Based on the information inside of the layer generates
    name that identifies layer inside of the graph.

    Parameters
    ----------
    layer : BaseLayer instance

    Returns
    -------
    str
        Layer's name
    """
    graph = layer.graph
    layer_class_name = layer.__class__.__name__

    if layer.layer_id is not None:
        layer_id = layer.layer_id

    else:
        graph_layers = graph.forward_graph.keys()
        layer_identifiers = []

        for graph_layer in graph_layers:
            if type(graph_layer) is type(layer) and graph_layer.layer_id:
                layer_identifiers.append(graph_layer.layer_id)

        layer_id = next_identifier(layer_identifiers)

    layer.layer_id = layer_id

    to_lowercase_regexp = re.compile(r'(?<!^)(?=[A-Z])')
    name_lower_case = to_lowercase_regexp.sub('-', layer_class_name).lower()
    layer_name = "{}-{}".format(name_lower_case, layer_id)

    return layer_name


def create_shared_parameter(value, name, shape):
    """
    Creates NN parameter as Theano shared variable.

    Parameters
    ----------
    value : array-like, Theano variable, scalar or Initializer
        Default value for the parameter.

    name : str
        Shared variable name.

    shape : tuple
        Parameter's shape.

    Returns
    -------
    Theano shared variable.
    """
    if isinstance(value, (T.sharedvar.SharedVariable, T.Variable)):
        return value

    if isinstance(value, init.Initializer):
        value = value.sample(shape)

    return theano.shared(value=asfloat(value), name=name, borrow=True)


class BaseLayer(BaseConnection, Configurable):
    """
    Base class for all layers.

    Parameters
    ----------
    name : str or None
        Layer's identifier. If name is equal to ``None`` than name
        will be generated automatically. Defaults to ``None``.

    Methods
    -------
    disable_training_state()
        Swith off trainig state.

    initialize()
        Set up important configurations related to the layer.

    Attributes
    ----------
    input_shape : tuple
        Layer's input shape.

    output_shape : tuple
        Layer's output shape.

    training_state : bool
        Defines whether layer in training state or not.

    layer_id : int
        Layer's identifier.

    parameters : dict
        Trainable parameters.

    graph : LayerGraph instance
        Graphs that stores all relations between layers.
    """
    name = Property(expected_type=six.string_types)

    # Stores global identifier index for each layer class
    global_identifiers_map = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls.global_identifiers_map:
            cls.global_identifiers_map[cls] = 1
        return super(BaseLayer, cls).__new__(cls)

    def __init__(self, *args, **options):
        super(BaseLayer, self).__init__(*args)

        self.parameters = {}
        self.updates = []
        self.input_shape_ = None

        cls = self.__class__
        self.layer_id = self.global_identifiers_map[cls]
        self.global_identifiers_map[cls] += 1

        self.graph.add_layer(self)

        Configurable.__init__(self, **options)

    def validate(self, input_shape):
        pass

    @property
    def input_shape(self):
        if self.input_shape_ is not None:
            return self.input_shape_

    @input_shape.setter
    def input_shape(self, value):
        self.validate(value)
        self.input_shape_ = value

    @property
    def output_shape(self):
        return self.input_shape

    def output(self, input_value):
        return input_value

    def initialize(self):
        self.parameters = {}

        if self.name is None:
            self.name = generate_layer_name(layer=self)

    def add_parameter(self, value, name, shape=None):
        theano_name = 'layer:{layer_name}/{parameter_name}'.format(
            layer_name=self.name,
            parameter_name=name.replace('_', '-')
        )

        parameter = create_shared_parameter(value, theano_name, shape)
        self.parameters[name] = parameter

        setattr(self, name, parameter)

    def __repr__(self):
        classname = self.__class__.__name__
        return '{name}()'.format(name=classname)


class ResidualConnection(BaseLayer):
    pass


class ParameterBasedLayer(BaseLayer):
    """
    Layer that creates weight and bias parameters.

    Parameters
    ----------
    size : int
        Layer's output size.

    weight : array-like, Theano variable, scalar or Initializer
        Defines layer's weights. Default initialization methods
        you can find :ref:`here <init-methods>`.
        Defaults to :class:`XavierNormal() <neupy.init.XavierNormal>`.

    bias : 1D array-like, Theano variable, scalar, Initializer or None
        Defines layer's bias.
        Default initialization methods you can find
        :ref:`here <init-methods>`. Defaults to
        :class:`XavierNormal() <neupy.init.XavierNormal>`.
        The ``None`` value excludes bias from the calculations and
        do not add it into parameters list.

    {BaseLayer.Parameters}

    Methods
    -------
    {BaseLayer.Methods}

    Attributes
    ----------
    {BaseLayer.Attributes}
    """
    size = IntProperty(minval=1)
    weight = ParameterProperty(default=init.XavierNormal())
    bias = ParameterProperty(default=init.XavierNormal(), allow_none=True)

    def __init__(self, size, **options):
        if size is not None:
            options['size'] = size
        super(ParameterBasedLayer, self).__init__(**options)

    @property
    def weight_shape(self):
        return as_tuple(self.input_shape, self.output_shape)

    @property
    def bias_shape(self):
        if self.bias is not None:
            return as_tuple(self.output_shape)

    def initialize(self):
        super(ParameterBasedLayer, self).initialize()

        self.add_parameter(value=self.weight, name='weight',
                           shape=self.weight_shape)

        if self.bias is not None:
            self.add_parameter(value=self.bias, name='bias',
                               shape=self.bias_shape)

    def __repr__(self):
        classname = self.__class__.__name__
        return '{name}({size})'.format(name=classname, size=self.size)


class ArrayShapeProperty(TypedListProperty):
    """
    Property that identifies array's shape.

    Parameters
    ----------
    {TypedListProperty.Parameters}
    """
    expected_type = (int, tuple)

    def validate(self, value):
        if not isinstance(value, int):
            super(ArrayShapeProperty, self).validate(value)

        elif value < 1:
            raise ValueError("Integer value is expected to be greater or "
                             " equal to one for the `{}` property, got {}"
                             "".format(self.name, value))


class Input(BaseLayer):
    """
    Input layer defines input's feature shape.

    Parameters
    ----------
    size : int, tuple or None
        Identifies input's feature shape.

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
    >>> input_layer = layers.Input(10)
    >>> input_layer
    Input(10)
    """
    size = ArrayShapeProperty(element_type=(int, type(None)))

    def __init__(self, size, **options):
        super(Input, self).__init__(size=size, **options)

        self.input_shape = as_tuple(self.size)
        self.initialize()

    @property
    def output_shape(self):
        return self.input_shape

    def output(self, input_value):
        return input_value

    def __repr__(self):
        classname = self.__class__.__name__
        return '{name}({size})'.format(name=classname, size=self.size)
