import re
import types
from collections import OrderedDict

import six
import tensorflow as tf

from neupy import init
from neupy.core.config import Configurable
from neupy.core.properties import Property
from neupy.utils import asfloat, class_method_name_scope
from neupy.layers.connections import BaseConnection


__all__ = ('BaseLayer', 'Identity')


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

    if classname.isupper():
        layer_name = classname.lower()
    else:
        layer_name = re.sub(r'(?<!^)(?=[A-Z][a-z_])', '-', classname)

    return "{}-{}".format(layer_name.lower(), layer_id)


def create_shared_parameter(value, name, shape, trainable):
    """
    Creates NN parameter as Tensorfow variable.

    Parameters
    ----------
    value : array-like, Tensorfow variable, scalar or Initializer
        Default value for the parameter.

    name : str
        Shared variable name.

    shape : tuple
        Parameter's shape.

    trainable : bool
        Whether parameter trainable by backpropagation.

    Returns
    -------
    Tensorfow variable.
    """
    if isinstance(value, tf.Variable):
        return value

    if isinstance(value, init.Initializer):
        value = value.sample(shape)

    return tf.Variable(
        asfloat(value),
        name=name,
        dtype=tf.float32,
        trainable=trainable,
    )


def initialize_layer(layer_class, kwargs, was_initialized):
    """
    We have a separate method for initialization, becase default
    __reduce__ functionality requires variables to be specified
    in order, which neupy doesn't support.
    """
    layer = layer_class(**kwargs)

    if was_initialized:
        layer.initialize()

    return layer


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
        Context manager that switches off trainig state.

    initialize()
        Set up important configurations related to the layer.

    Attributes
    ----------
    input_shape : tuple
        Returns layer's input shape in the form of a tuple. Shape will
        not include batch size dimension.

    output_shape : tuple
        Returns layer's output shape in the form of a tuple. Shape will
        not include batch size dimension.

    training_state : bool
        Defines whether layer in training state or not. Training state
        will enable some operations inside of the layers that won't
        work otherwise.

    parameters : dict
        Parameters that networks uses during propagation. It might include
        trainable and non-trainable parameters.

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

        self.output = types.MethodType(
            class_method_name_scope(self.output), self)

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
    def input_shape(self, shape):
        self.validate(shape)
        self.input_shape_ = shape

    @property
    def output_shape(self):
        return self.input_shape

    def output(self, input_value):
        return input_value

    def add_parameter(self, value, name, shape=None, trainable=True):
        layer_name = 'layer/{layer_name}/{parameter_name}'.format(
            layer_name=self.name,
            parameter_name=name.replace('_', '-'))

        self.parameters[name] = create_shared_parameter(
            value, layer_name, shape, trainable)

        setattr(self, name, self.parameters[name])
        return self.parameters[name]

    def __repr__(self):
        classname = self.__class__.__name__
        return '{name}()'.format(name=classname)

    def __reduce__(self):
        parameters = self.get_params()
        return (initialize_layer, (
            self.__class__, parameters, self.initialized))


class Identity(BaseLayer):
    """
    Residual skip connection.
    """
