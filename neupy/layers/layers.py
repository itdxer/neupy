import numpy as np
import theano
import theano.tensor as T

from neupy.utils import asfloat
from neupy.core.properties import (TypedListProperty, ArrayProperty,
                                   ChoiceProperty, ProperFractionProperty,
                                   IntProperty)
from .base import BaseLayer
from .utils import XAVIER_NORMAL, VALID_INIT_METHODS, generate_weight
from .activations import relu, step_function


__all__ = ('Layer', 'Linear', 'Sigmoid', 'HardSigmoid', 'Step', 'Tanh',
           'Relu', 'Softplus', 'Softmax', 'Dropout')


theano_shared_class = T.sharedvar.TensorSharedVariable


class SharedArrayProperty(ArrayProperty):
    """ In addition to Numpy arrays and matrix property support also
    Theano shared variables.

    Parameters
    ----------
    {BaseProperty.default}
    {BaseProperty.required}
    """
    expected_type = (np.matrix, np.ndarray, theano_shared_class,
                     T.TensorVariable)


class Layer(BaseLayer):
    """ Base class for input and hidden layers.

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
    size = IntProperty(required=True, minval=1)
    weight = SharedArrayProperty(default=None)
    bias = SharedArrayProperty(default=None)
    bounds = TypedListProperty(default=(0, 1), element_type=(int, float))
    init_method = ChoiceProperty(default=XAVIER_NORMAL,
                                 choices=VALID_INIT_METHODS)

    def __init__(self, size, **options):
        options['size'] = size
        super(Layer, self).__init__(**options)

    def initialize(self):
        output_size = self.relate_to_layer.size
        weight = self.weight
        bias = self.bias
        self.step = None

        if self.relate_from_layer is not None:
            self.layer_id = self.relate_from_layer.layer_id + 1

        self.weight_shape = (self.size, output_size)
        self.bias_shape = (output_size,)

        # TODO: This part looks ugly, should find a different way.
        if not isinstance(weight, theano_shared_class):
            if weight is None:
                weight = generate_weight(self.weight_shape, self.bounds,
                                         self.init_method)

            self.weight = theano.shared(value=asfloat(weight),
                                        name='weight_{}'.format(self.layer_id),
                                        borrow=True)

        if not isinstance(bias, theano_shared_class):
            if bias is None:
                bias = generate_weight(self.bias_shape, self.bounds,
                                       self.init_method)

            self.bias = theano.shared(value=asfloat(bias),
                                      name='bias_{}'.format(self.layer_id),
                                      borrow=True)

        self.parameters = [self.weight, self.bias]

    def output(self, input_value):
        summated = T.dot(input_value, self.weight) + self.bias
        return self.activation_function(summated)

    def __repr__(self):
        return '{name}({size})'.format(name=self.__class__.__name__,
                                       size=self.size)


class Linear(Layer):
    """ The layer with the linear activation function.

    Parameters
    ----------
    {Layer.size}
    {Layer.weight}
    {Layer.bias}
    {Layer.init_method}
    {Layer.bounds}
    """
    activation_function = (lambda x: x)


class Sigmoid(Layer):
    """ The layer with the sigmoid activation function.

    Parameters
    ----------
    {Layer.size}
    {Layer.weight}
    {Layer.bias}
    {Layer.init_method}
    {Layer.bounds}
    """
    activation_function = T.nnet.sigmoid


class HardSigmoid(Layer):
    """ The layer with the hard sigmoid activation function.

    Parameters
    ----------
    {Layer.size}
    {Layer.weight}
    {Layer.bias}
    {Layer.init_method}
    {Layer.bounds}
    """
    activation_function = T.nnet.hard_sigmoid


class Step(Layer):
    """ The layer with the the step activation function.

    Parameters
    ----------
    {Layer.size}
    {Layer.weight}
    {Layer.bias}
    {Layer.init_method}
    {Layer.bounds}
    """
    activation_function = step_function


class Tanh(Layer):
    """ The layer with the `tanh` activation function.

    Parameters
    ----------
    {Layer.size}
    {Layer.weight}
    {Layer.bias}
    {Layer.init_method}
    {Layer.bounds}
    """
    activation_function = T.tanh


class Relu(Layer):
    """ The layer with the rectifier activation function.

    Parameters
    ----------
    {Layer.size}
    {Layer.weight}
    {Layer.bias}
    {Layer.init_method}
    {Layer.bounds}
    """
    activation_function = relu


class Softplus(Layer):
    """ The layer with the softplus activation function.

    Parameters
    ----------
    {Layer.size}
    {Layer.weight}
    {Layer.bias}
    {Layer.init_method}
    {Layer.bounds}
    """
    activation_function = T.nnet.softplus


class Softmax(Layer):
    """ The layer with the softmax activation function.

    Parameters
    ----------
    {Layer.size}
    {Layer.weight}
    {Layer.bias}
    {Layer.init_method}
    {Layer.bounds}
    """
    activation_function = T.nnet.softmax


class Dropout(BaseLayer):
    """ Dropout layer

    Parameters
    ----------
    proba : float
        Fraction of the input units to drop. Value needs to be
        between 0 and 1.
    """
    proba = ProperFractionProperty(required=True)

    def __init__(self, proba, **options):
        options['proba'] = proba
        super(Dropout, self).__init__(**options)

    @property
    def size(self):
        return self.relate_to_layer.size

    def initialize(self):
        pass

    def output(self, input_value):
        # Use NumPy seed to make Theano code easely reproducible
        max_possible_seed = 4e9
        seed = np.random.randint(max_possible_seed)
        theano_random = T.shared_randomstreams.RandomStreams(seed)

        proba = (1.0 - self.proba)
        mask = theano_random.binomial(n=1, p=proba,
                                      size=input_value.shape,
                                      dtype=input_value.dtype)
        return (mask * input_value) / proba

    def __repr__(self):
        return "{name}(proba={proba})".format(name=self.__class__.__name__,
                                              proba=self.proba)
