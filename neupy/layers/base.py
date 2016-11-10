import re

import six
import theano
import theano.tensor as T

from neupy import init
from neupy.utils import asfloat, as_tuple
from neupy.core.config import Configurable
from neupy.core.properties import (TypedListProperty, IntProperty, Property,
                                   ParameterProperty)
from neupy.layers.connections import ChainConnection


__all__ = ('BaseLayer', 'ParameterBasedLayer', 'Input')


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

    elif graph is not None:
        graph_layers = graph.forward_graph.keys()
        layer_identifiers = []

        for graph_layer in graph_layers:
            if type(graph_layer) is type(layer) and graph_layer.layer_id:
                layer_identifiers.append(graph_layer.layer_id)

        layer_id = 1 + max(layer_identifiers, default=0)

    else:
        layer_id = 1

    layer.layer_id = layer_id

    to_lowercase_regexp = re.compile(r'(?<!^)(?=[A-Z])')
    name_lower_case = to_lowercase_regexp.sub('-', layer_class_name).lower()
    layer_name = "{}-{}".format(name_lower_case, layer_id)

    return layer_name


class BaseLayer(ChainConnection, Configurable):
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
    output_shape : tuple
    training_state : bool
        Defines whether layer in training state or not.
    layer_id : int
        Layer's identifier.
    parameters : list
        List of layer's parameters.
    graph : LayerGraph instance or None
    """
    name = Property(expected_type=six.string_types)

    def __init__(self, *args, **options):
        super(BaseLayer, self).__init__(*args)

        self.parameters = []
        self.updates = []
        self.layer_id = None
        self.input_shape_ = None

        Configurable.__init__(self, **options)

    @property
    def input_shape(self):
        if self.input_shape_ is not None:
            return self.input_shape_

    @input_shape.setter
    def input_shape(self, value):
        self.input_shape_ = value

    @property
    def output_shape(self):
        return self.input_shape

    def output(self, input_value):
        return input_value

    def initialize(self):
        self.parameters = []

        if self.name is None:
            self.name = generate_layer_name(layer=self)

    def add_parameter(self, value, name, shape):
        theano_name = '{layer_name}/{parameter_name}'.format(
            layer_name=self.name,
            parameter_name=name
        )

        parameter = create_shared_parameter(value, theano_name, shape)
        self.parameters.append(parameter)

        setattr(self, name, parameter)

    def __repr__(self):
        classname = self.__class__.__name__
        return '{name}()'.format(name=classname)


def create_shared_parameter(value, name, shape):
    """
    Creates NN parameter as Theano shared variable.

    Parameters
    ----------
    value : array-like, Theano variable, scalar or Initializer
        Default value for the parameter.
    name : str
        Sahred variable name.
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
        Defaults to :class:`XavierNormal() <neupy.core.init.XavierNormal>`.
    bias : 1D array-like, Theano variable, scalar or Initializer
        Defines layer's bias. Default initialization methods
        you can find :ref:`here <init-methods>`.
        Defaults to :class:`XavierNormal() <neupy.core.init.XavierNormal>`.

    Methods
    -------
    {BaseLayer.Methods}

    Attributes
    ----------
    {BaseLayer.Attributes}
    """
    size = IntProperty(minval=1)
    weight = ParameterProperty(default=init.XavierNormal())
    bias = ParameterProperty(default=init.XavierNormal())

    def __init__(self, size, **options):
        if size is not None:
            options['size'] = size
        super(ParameterBasedLayer, self).__init__(**options)

    @property
    def weight_shape(self):
        return as_tuple(self.input_shape, self.output_shape)

    @property
    def bias_shape(self):
        return as_tuple(self.output_shape)

    def initialize(self):
        super(ParameterBasedLayer, self).initialize()
        self.add_parameter(value=self.weight, name='weight',
                           shape=self.weight_shape)
        self.add_parameter(value=self.bias, name='bias',
                           shape=self.bias_shape)

    def __repr__(self):
        classname = self.__class__.__name__
        return '{name}({size})'.format(name=classname, size=self.size)


class ArrayShapeProperty(TypedListProperty):
    """
    Property that identifies array's shape.
    """
    expected_type = (int, tuple, type(None))

    def validate(self, value):
        if isinstance(value, int):
            if value < 1:
                raise ValueError("Integer value is expected to be greater or "
                                 " equal to one for the `{}` property, got {}"
                                 "".format(self.name, value))
        elif value is not None:
            super(ArrayShapeProperty, self).validate(value)


class Input(BaseLayer):
    """
    Input layer. It identifies feature shape/size for the
    input value. Especially useful in the CNN.

    Parameters
    ----------
    size : int, tuple or None
        Identifies input data shape size. ``None`` means that network
        doesn't have input feature with fixed size.
        Defaults to ``None``.
    {BaseLayer.Parameters}

    Methods
    -------
    {BaseLayer.Methods}

    Attributes
    ----------
    {BaseLayer.Attributes}
    """
    size = ArrayShapeProperty(element_type=(int, type(None)))

    def __init__(self, size, **options):
        super(Input, self).__init__(size=size, **options)
        self.input_shape = as_tuple(self.size)

    @property
    def output_shape(self):
        return self.input_shape

    def output(self, input_value):
        return input_value

    def __repr__(self):
        classname = self.__class__.__name__
        return '{name}({size})'.format(name=classname, size=self.size)
