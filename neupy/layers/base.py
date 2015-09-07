from six import with_metaclass
from numpy import dot
from numpy.random import randn

from neupy.core.config import ConfigMeta, BaseConfigurable
from neupy.core.properties import (IntProperty, NumberBoundProperty,
                                   ArrayProperty, ChoiceProperty)
from neupy.network.connections import ChainConnection
from neupy.network.utils import add_bias_column
from neupy.layers.utils import random_orthogonal, random_bounded


__all__ = ('BaseLayer',)


GAUSSIAN = 'gauss'
BOUNDED = 'bounded'
ORTHOGONAL = 'ortho'


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
        Define your layer weights. `None` means that your weights will be
        generate randomly dependence on property `init_method`.
        `None` by default.
    init_method : {'gauss', 'bounded', 'ortho'}
        Weight initialization method.
        `gauss` will generate random weights dependence on Standard
        Normal Distribution.
        `bounded` generate uniform random weghts in initialized bounds.
        `ortho` generate random orthogonal matrix.
    random_weight_bound : tuple of two int
        Available only for `init_method` eqaul to `bounded`, defaults
        to `(0, 1)`.
    """
    shared_docs = {'layer_params': __layer_params}

    input_size = IntProperty()
    weight = ArrayProperty(default=None)
    random_weight_bound = NumberBoundProperty(default=(0, 1))
    init_method = ChoiceProperty(default=GAUSSIAN,
                                 choices=[GAUSSIAN, BOUNDED, ORTHOGONAL])

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

        # Initialize default options
        BaseConfigurable.__init__(self, **options)

    def relate_to(self, right_layer):
        self.relate_to_layer = right_layer

    def initialize(self, with_bias=False):
        self.use_bias = with_bias
        size = self.input_size + self.use_bias
        self.size = (size, self.relate_to_layer.input_size)
        self.weight = self._init_weight()

    # --------------- Weights manipulations --------------- #

    def _init_weight(self):
        if self.weight is not None:
            return self.weight

        init_method = self.init_method

        if init_method == GAUSSIAN:
            return randn(*self.size)

        elif init_method == BOUNDED:
            return random_bounded(self.size, *self.random_weight_bound)

        elif init_method == ORTHOGONAL:
            return random_orthogonal(self.size)

    @property
    def weight_without_bias(self):
        if self.use_bias:
            return self.weight[1:, :]
        return self.weight

    # --------------- Layer operations --------------- #

    def summator(self, input_value):
        return dot(input_value, self.weight)

    def output(self, input_value):
        input_data = self.preformat_input(input_value)
        summated = self.summator(input_data)
        return self.activation_function(summated)

    def preformat_input(self, input_data):
        if self.use_bias:
            input_data = add_bias_column(input_data)
        return input_data

    def __repr__(self):
        return '{name}({size})'.format(name=self.__class__.__name__,
                                       size=self.input_size)
