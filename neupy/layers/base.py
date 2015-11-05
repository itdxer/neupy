import theano
import theano.tensor as T
from six import with_metaclass

from neupy.utils import asfloat
from neupy.core.config import ConfigMeta, BaseConfigurable
from neupy.core.properties import (IntProperty, NumberBoundProperty,
                                   ArrayProperty, ChoiceProperty)
from neupy.network.connections import ChainConnection
from neupy.layers.utils import GAUSSIAN, VALID_INIT_METHODS, generate_weight


__all__ = ('BaseLayer',)


class SharedArrayProperty(ArrayProperty):
    expected_type = (ArrayProperty.expected_type,
                     theano.tensor.sharedvar.TensorSharedVariable)


class LayerMeta(ConfigMeta):
    def __new__(cls, clsname, bases, attrs):
        if 'activation_function' in attrs:
            # Override `activation_function` in `staticmethod` by default.
            # Python 2 translate any assigned function as class method
            # and try call it with with argument `self` which broke
            # logic. For this reason we try make it static.
            attrs['activation_function'] = staticmethod(
                attrs['activation_function']
            )
        return super(LayerMeta, cls).__new__(cls, clsname, bases, attrs)


class BaseLayer(with_metaclass(LayerMeta, ChainConnection, BaseConfigurable)):
    """ Base class for all layers.

    Parameters
    ----------
    {layer_params}
    """
    __layer_params = """input_size : int
        Layer input size.
    weight : 2D array-like or None
        Define your layer weights. ``None`` means that your weights will be
        generate randomly dependence on property ``init_method``.
        ``None`` by default.
    init_method : {'gauss', 'bounded', 'ortho'}
        Weight initialization method.
        ``gauss`` will generate random weights from Standard Normal
        Distribution.
        ``bounded`` generate random weights from Uniform distribution.
        ``ortho`` generate random orthogonal matrix.
        Defaults to ``gauss``.
    bounds : tuple of two float
        Available only for ``init_method`` eqaul to ``bounded``.  Value
        identify minimum and maximum possible value in random weights.
        Defaults to ``(0, 1)``.
    """
    shared_docs = {'layer_params': __layer_params}

    input_size = IntProperty()
    weight = SharedArrayProperty(default=None)
    bias = SharedArrayProperty(default=None)
    bounds = NumberBoundProperty(default=(0, 1))
    init_method = ChoiceProperty(default=GAUSSIAN, choices=VALID_INIT_METHODS)

    def __init__(self, input_size, **options):
        super(BaseLayer, self).__init__()

        self.input_size = input_size
        self.use_bias = False

        # Default variables which will change after initialization
        self.relate_to_layer = None
        self.size = None

        # If you will set class method function variable, python understend
        # that this is new class method and will call it with `self`
        # first parameter.
        if hasattr(self.__class__, 'activation_function'):
            self.activation_function = self.__class__.activation_function

        BaseConfigurable.__init__(self, **options)

    def relate_to(self, right_layer):
        self.relate_to_layer = right_layer

    def initialize(self, with_bias=False):
        self.use_bias = with_bias
        output_size = self.relate_to_layer.input_size

        weight = self.weight
        bias = self.bias

        if weight is None:
            weight_shape = (self.input_size, output_size)
            weight = generate_weight(weight_shape, self.bounds,
                                     self.init_method)

        self.weight = theano.shared(value=asfloat(weight), name='w',
                                    borrow=True)

        if with_bias:
            if bias is None:
                bias_shape = (output_size,)
                bias = generate_weight(bias_shape, self.bounds,
                                       self.init_method)

            self.bias = theano.shared(value=asfloat(bias), name='b',
                                      borrow=True)

    def output(self, input_value):
        summated = T.dot(input_value, self.weight)
        if self.use_bias:
            summated += self.bias
        return self.activation_function(summated)

    def __repr__(self):
        return '{name}({size})'.format(name=self.__class__.__name__,
                                       size=self.input_size)
