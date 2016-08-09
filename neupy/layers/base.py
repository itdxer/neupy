import theano
import theano.tensor as T

from neupy.utils import asfloat, as_tuple
from neupy.core.config import Configurable
from neupy.core.properties import (TypedListProperty, IntProperty,
                                   ParameterProperty)
from neupy.layers.connections import ChainConnection
from neupy.core.init import XavierNormal, Initializer


__all__ = ('BaseLayer', 'ParameterBasedLayer', 'Input')


class BaseLayer(ChainConnection, Configurable):
    """
    Base class for all layers.

    Methods
    -------
    initialize()
        Set up important configurations related to the layer.
    relate_to(right_layer)
        Connect current layer with the next one.
    disable_training_state()
        Swith off trainig state.

    Attributes
    ----------
    training_state : bool
        Defines whether layer in training state or not.
    layer_id : int
        Layer's identifier.
    parameters : list
        List of layer's parameters.
    relate_to_layer : BaseLayer or None
    relate_from_layer : BaseLayer or None
    """
    def __init__(self, *args, **options):
        super(BaseLayer, self).__init__(*args)

        self.parameters = []

        self.relate_to_layer = None
        self.relate_from_layer = None
        self.layer_id = 1
        self.updates = []

        Configurable.__init__(self, **options)

    @property
    def input_shape(self):
        if self.relate_from_layer is not None:
            return self.relate_from_layer.output_shape

    @property
    def output_shape(self):
        return self.input_shape

    def output(self, input_value):
        return input_value

    def initialize(self):
        if self.relate_from_layer is not None:
            self.layer_id = self.relate_from_layer.layer_id + 1

    def relate_to(self, right_layer):
        self.relate_to_layer = right_layer
        right_layer.relate_from_layer = self

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

    if isinstance(value, Initializer):
        value = value.sample(shape)

    return theano.shared(value=asfloat(value), name=name, borrow=True)


class ParameterBasedLayer(BaseLayer):
    """
    Layer that creates weight and bias parameters.

    Parameters
    ----------
    size : int
        Layer input size.
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
    weight = ParameterProperty(default=XavierNormal())
    bias = ParameterProperty(default=XavierNormal())

    def __init__(self, size, **options):
        if size is not None:
            options['size'] = size
        super(ParameterBasedLayer, self).__init__(**options)

    @property
    def output_shape(self):
        return as_tuple(self.relate_to_layer.size)

    @property
    def weight_shape(self):
        return as_tuple(self.input_shape, self.output_shape)

    @property
    def bias_shape(self):
        return as_tuple(self.output_shape)

    def initialize(self):
        super(ParameterBasedLayer, self).initialize()

        self.weight = create_shared_parameter(
            value=self.weight,
            name='weight_{}'.format(self.layer_id),
            shape=self.weight_shape
        )
        self.bias = create_shared_parameter(
            value=self.bias,
            name='bias_{}'.format(self.layer_id),
            shape=self.bias_shape
        )
        self.parameters = [self.weight, self.bias]

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

    Methods
    -------
    {BaseLayer.Methods}

    Attributes
    ----------
    {BaseLayer.Attributes}
    """
    size = ArrayShapeProperty()

    def __init__(self, size, **options):
        super(Input, self).__init__(size=size, **options)

    @property
    def input_shape(self):
        return as_tuple(self.size)

    @property
    def output_shape(self):
        return self.input_shape

    def output(self, input_value):
        return input_value

    def __repr__(self):
        classname = self.__class__.__name__
        return '{name}({size})'.format(name=classname, size=self.size)
