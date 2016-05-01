import numpy as np
import theano
import theano.tensor as T

from neupy.utils import asfloat
from neupy.core.config import Configurable
from neupy.core.properties import (TypedListProperty, ArrayProperty,
                                   ChoiceProperty, IntProperty)
from neupy.layers.connections import ChainConnection
from .utils import XAVIER_NORMAL, VALID_INIT_METHODS, generate_weight


__all__ = ('BaseLayer', 'ParameterBasedLayer')


class BaseLayer(ChainConnection, Configurable):
    """ Base class for all layers.
    """
    def __init__(self, *args, **options):
        super(BaseLayer, self).__init__()

        self.parameters = []

        # Default variables which will change after initialization
        self.relate_to_layer = None
        self.relate_from_layer = None
        self.layer_id = 1

        Configurable.__init__(self, **options)

    def initialize(self):
        if self.relate_from_layer is not None:
            self.layer_id = self.relate_from_layer.layer_id + 1

    def relate_to(self, right_layer):
        self.relate_to_layer = right_layer
        right_layer.relate_from_layer = self

    def __repr__(self):
        classname = self.__class__.__name__
        return '{name}()'.format(name=classname)


def create_shared_parameter(value, name, shape, init_method, bounds):
    """ Creates NN parameter as Theano shared variable.

    Parameters
    ----------
    value : array-like, theano shared variable or None
        Default value for the parameter. If value eqaul to ``None``
        parameter will be created bsaed on the ``init_method`` value.
    name : str
        Sahred variable name.
    shape : tuple
        Parameter shape.
    init_method : str
        Weight initialization procedure name.
    bounds : tuple
        Specific parameter for the one of the ``init_method``
        argument.

    Returns
    -------
    Theano shared variable.
    """
    if isinstance(value, (T.sharedvar.SharedVariable, T.Variable)):
        return value

    if value is None:
        value = generate_weight(shape, bounds, init_method)

    return theano.shared(value=asfloat(value), name=name, borrow=True)


class SharedArrayProperty(ArrayProperty):
    """ In addition to Numpy arrays and matrix property support also
    Theano shared variables.

    Parameters
    ----------
    {BaseProperty.default}
    {BaseProperty.required}
    """
    expected_type = (np.matrix, np.ndarray,
                     T.sharedvar.SharedVariable,
                     T.Variable)


class ParameterBasedLayer(BaseLayer):
    """ Layer that creates weight and bias parameters.

    Parameters
    ----------
    size : int
        Layer input size.
    weight : 2D array-like or None
        Define your layer weights. ``None`` means that your weights will be
        generate randomly dependence on property ``init_method``.
        ``None`` by default.
    bias : 1D array-like or None
        Define your layer bias. ``None`` means that your weights will be
        generate randomly dependence on property ``init_method``.
    init_method : {{'bounded', 'normal', 'ortho', 'xavier_normal',\
    'xavier_uniform', 'he_normal', 'he_uniform'}}
        Weight initialization method. Defaults to ``xavier_normal``.

        * ``normal`` will generate random weights from normal distribution \
        with standard deviation equal to ``0.01``.

        * ``bounded`` generate random weights from Uniform distribution.

        * ``ortho`` generate random orthogonal matrix.

        * ``xavier_normal`` generate random matrix from normal distrubtion \
        where variance equal to :math:`\\frac{{2}}{{fan_{{in}} + \
        fan_{{out}}}}`. Where :math:`fan_{{in}}` is a number of \
        layer input units and :math:`fan_{{out}}` - number of layer \
        output units.

        * ``xavier_uniform`` generate random matrix from uniform \
        distribution \ where :math:`w_{{ij}} \in \
        [-\\sqrt{{\\frac{{6}}{{fan_{{in}} + fan_{{out}}}}}}, \
        \\sqrt{{\\frac{{6}}{{fan_{{in}} + fan_{{out}}}}}}`].

        * ``he_normal`` generate random matrix from normal distrubtion \
        where variance equal to :math:`\\frac{{2}}{{fan_{{in}}}}`. \
        Where :math:`fan_{{in}}` is a number of layer input units.

        * ``he_uniform`` generate random matrix from uniformal \
        distribution where :math:`w_{{ij}} \in [\
        -\\sqrt{{\\frac{{6}}{{fan_{{in}}}}}}, \
        \\sqrt{{\\frac{{6}}{{fan_{{in}}}}}}]`

    bounds : tuple of two float
        Available only for ``init_method`` equal to ``bounded``.  Value
        identify minimum and maximum possible value in random weights.
        Defaults to ``(0, 1)``.
    """
    size = IntProperty(minval=1)
    weight = SharedArrayProperty(default=None)
    bias = SharedArrayProperty(default=None)
    bounds = TypedListProperty(default=(0, 1), element_type=(int, float))
    init_method = ChoiceProperty(default=XAVIER_NORMAL,
                                 choices=VALID_INIT_METHODS)

    def __init__(self, size, **options):
        if size is not None:
            options['size'] = size
        super(ParameterBasedLayer, self).__init__(**options)

    def weight_shape(self):
        output_size = self.relate_to_layer.size
        return (self.size, output_size)

    def bias_shape(self):
        output_size = self.relate_to_layer.size
        return (output_size,)

    def initialize(self):
        super(ParameterBasedLayer, self).initialize()

        self.weight = create_shared_parameter(
            value=self.weight,
            name='weight_{}'.format(self.layer_id),
            shape=self.weight_shape(),
            bounds=self.bounds,
            init_method=self.init_method,
        )
        self.bias = create_shared_parameter(
            value=self.bias,
            name='bias_{}'.format(self.layer_id),
            shape=self.bias_shape(),
            bounds=self.bounds,
            init_method=self.init_method,
        )
        self.parameters = [self.weight, self.bias]

    def __repr__(self):
        classname = self.__class__.__name__
        return '{name}({size})'.format(name=classname, size=self.size)
