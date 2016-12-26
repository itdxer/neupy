import re
from collections import OrderedDict

import six
import theano
import theano.tensor as T

from neupy import init
from neupy.utils import asfloat, as_tuple
from neupy.core.config import Configurable
from neupy.core.properties import ParameterProperty, IntProperty, Property
from neupy.layers.connections import BaseConnection


__all__ = ('BaseLayer', 'ParameterBasedLayer', 'ResidualConnection')


def generate_layer_name(layer):
    """
    Generates unique name for layer.

    Parameters
    ----------
    layer : BaseLayer

    Returns
    -------
    str
    """
    cls = layer.__class__

    layer_id = cls.global_identifiers_map[cls]
    cls.global_identifiers_map[cls] += 1

    classname = cls.__name__
    layer_name = re.sub(r'(?<!^)(?=[A-Z])', '-', classname)

    return "{}-{}".format(layer_name.lower(), layer_id)


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

        self.updates = []
        self.parameters = OrderedDict()
        self.name = generate_layer_name(layer=self)
        self.input_shape_ = None

        self.graph.add_layer(self)

        Configurable.__init__(self, **options)

    def validate(self, input_shape):
        """
        Validate input shape value before assigning it.

        Parameters
        ----------
        input_shape : tuple with int
        """

    @property
    def input_shape(self):
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

    def add_parameter(self, value, name, shape=None, trainable=True):
        theano_name = 'layer:{layer_name}/{parameter_name}'.format(
            layer_name=self.name,
            parameter_name=name.replace('_', '-'))

        parameter = create_shared_parameter(value, theano_name, shape)
        parameter.trainable = trainable

        self.parameters[name] = parameter

        setattr(self, name, parameter)

    def __repr__(self):
        classname = self.__class__.__name__
        return '{name}()'.format(name=classname)


class ResidualConnection(BaseLayer):
    """
    Residual skip connection.
    """


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
        :class:`Constant(0) <neupy.init.Constant>`.
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
    bias = ParameterProperty(default=init.Constant(value=0), allow_none=True)

    def __init__(self, size, **options):
        super(ParameterBasedLayer, self).__init__(size=size, **options)

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
                           shape=self.weight_shape, trainable=True)

        if self.bias is not None:
            self.add_parameter(value=self.bias, name='bias',
                               shape=self.bias_shape, trainable=True)

    def __repr__(self):
        classname = self.__class__.__name__
        return '{name}({size})'.format(name=classname, size=self.size)
