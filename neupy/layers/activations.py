import theano.tensor as T

from neupy.utils import asfloat, as_tuple
from neupy.core.properties import (NumberProperty, TypedListProperty,
                                   ParameterProperty)
from neupy.core.init import Initializer, Constant
from .utils import dimshuffle
from .base import ParameterBasedLayer, create_shared_parameter


__all__ = ('ActivationLayer', 'Linear', 'Sigmoid', 'HardSigmoid', 'Step',
           'Tanh', 'Relu', 'Softplus', 'Softmax', 'Elu', 'PRelu')


class ActivationLayer(ParameterBasedLayer):
    """
    Base class for the layers based on the activation
    functions.

    Parameters
    ----------
    size : int or None
        Layer input size. ``None`` means that layer will not create
        parameters and will return only activation function
        output for the specified input value.
    {ParameterBasedLayer.weight}
    {ParameterBasedLayer.bias}

    Methods
    -------
    {ParameterBasedLayer.Methods}

    Attributes
    ----------
    {ParameterBasedLayer.Attributes}
    """
    def __init__(self, size=None, **options):
        super(ActivationLayer, self).__init__(size, **options)

    @property
    def output_shape(self):
        if self.size is not None:
            return as_tuple(self.size)
        return self.input_shape

    def initialize(self):
        if self.size is not None:
            super(ActivationLayer, self).initialize()

    def output(self, input_value):
        if self.size is not None:
            input_value = T.dot(input_value, self.weight) + self.bias
        return self.activation_function(input_value)

    def __repr__(self):
        if self.size is None:
            return super(ParameterBasedLayer, self).__repr__()
        return super(ActivationLayer, self).__repr__()


class Linear(ActivationLayer):
    """
    The layer with the linear activation function.

    Parameters
    ----------
    {ActivationLayer.Parameters}

    Methods
    -------
    {ActivationLayer.Methods}

    Attributes
    ----------
    {ActivationLayer.Attributes}
    """
    def activation_function(self, input_value):
        return input_value


class Sigmoid(ActivationLayer):
    """
    The layer with the sigmoid activation function.

    Parameters
    ----------
    {ActivationLayer.Parameters}

    Methods
    -------
    {ActivationLayer.Methods}

    Attributes
    ----------
    {ActivationLayer.Attributes}
    """
    def activation_function(self, input_value):
        return T.nnet.sigmoid(input_value)


class HardSigmoid(ActivationLayer):
    """
    The layer with the hard sigmoid activation function.

    Parameters
    ----------
    {ActivationLayer.Parameters}

    Methods
    -------
    {ActivationLayer.Methods}

    Attributes
    ----------
    {ActivationLayer.Attributes}
    """
    def activation_function(self, input_value):
        return T.nnet.hard_sigmoid(input_value)


class Step(ActivationLayer):
    """
    The layer with the the step activation function.

    Parameters
    ----------
    {ActivationLayer.Parameters}

    Methods
    -------
    {ActivationLayer.Methods}

    Attributes
    ----------
    {ActivationLayer.Attributes}
    """
    def activation_function(self, input_value):
        return T.gt(input_value, 0)


class Tanh(ActivationLayer):
    """
    The layer with the `tanh` activation function.

    Parameters
    ----------
    {ActivationLayer.Parameters}

    Methods
    -------
    {ActivationLayer.Methods}

    Attributes
    ----------
    {ActivationLayer.Attributes}
    """
    def activation_function(self, input_value):
        return T.tanh(input_value)


class Relu(ActivationLayer):
    """
    The layer with the rectifier (ReLu) activation function.

    Parameters
    ----------
    alpha : float
        Alpha parameter defines the decreasing rate
        for the negative values. If ``alpha``
        is non-zero value then layer behave like a
        leaky ReLu. Defaults to ``0``.
    {ActivationLayer.Parameters}

    Methods
    -------
    {ActivationLayer.Methods}

    Attributes
    ----------
    {ActivationLayer.Attributes}
    """
    alpha = NumberProperty(default=0, minval=0)

    def activation_function(self, input_value):
        alpha = asfloat(self.alpha)
        return T.nnet.relu(input_value, alpha)


class Softplus(ActivationLayer):
    """
    The layer with the softplus activation function.

    Parameters
    ----------
    {ActivationLayer.Parameters}

    Methods
    -------
    {ActivationLayer.Methods}

    Attributes
    ----------
    {ActivationLayer.Attributes}
    """

    def activation_function(self, input_value):
        return T.nnet.softplus(input_value)


class Softmax(ActivationLayer):
    """
    The layer with the softmax activation function.

    Parameters
    ----------
    {ActivationLayer.Parameters}

    Methods
    -------
    {ActivationLayer.Methods}

    Attributes
    ----------
    {ActivationLayer.Attributes}
    """

    def activation_function(self, input_value):
        return T.nnet.softmax(input_value)


class Elu(ActivationLayer):
    """
    The layer with the exponensial linear unit (ELU)
    activation function.

    Parameters
    ----------
    alpha : float
        Alpha parameter defines the decreasing exponensial
        rate for the negative values. Defaults to ``1``.
    {ActivationLayer.Parameters}

    Methods
    -------
    {ActivationLayer.Methods}

    Attributes
    ----------
    {ActivationLayer.Attributes}

    References
    ----------
    .. [1] http://arxiv.org/pdf/1511.07289v3.pdf
    """
    alpha = NumberProperty(default=1, minval=0)

    def activation_function(self, input_value):
        alpha = asfloat(self.alpha)
        return T.nnet.elu(input_value, alpha)


class AxesProperty(TypedListProperty):
    """
    Property defines axes parameter.

    Parameters
    ----------
    {TypedListProperty.n_elements}
    {TypedListProperty.element_type}
    {BaseProperty.default}
    {BaseProperty.required}
    """
    def __set__(self, instance, value):
        if isinstance(value, int):
            value = (value,)
        super(AxesProperty, self).__set__(instance, value)

    def validate(self, value):
        super(AxesProperty, self).validate(value)

        if any(element < 0 for element in value):
            raise ValueError("Axes property is allowed only "
                             "non-negative axis.")


class PRelu(ActivationLayer):
    """
    The layer with the parametrized ReLu activation
    function.

    Parameters
    ----------
    alpha_axes : int or tuple
        Axes that will not include unique alpha parameter.
        Single integer value defines the same as a tuple with one value.
        Defaults to ``1``.
    alpha : array-like, Theano shared variable, scalar or Initializer
        Alpha parameter per each non-shared axis for the ReLu.
        Scalar value means that each element in the tensor will be
        equal to the specified value.
        Default initialization methods you can find
        :ref:`here <init-methods>`.
        Defaults to ``Constant(value=0.25)``.
    {ActivationLayer.Parameters}

    Methods
    -------
    {ActivationLayer.Methods}

    Attributes
    ----------
    {ActivationLayer.Attributes}

    References
    ----------
    .. [1] https://arxiv.org/pdf/1502.01852v1.pdf
    """
    alpha_axes = AxesProperty(default=1)
    alpha = ParameterProperty(default=Constant(value=0.25))

    def initialize(self):
        super(PRelu, self).initialize()

        alpha = self.alpha
        alpha_axes = self.alpha_axes
        output_shape = self.output_shape

        if 0 in alpha_axes:
            raise ValueError("Cannot specify alpha per input sample.")

        if max(alpha_axes) > len(output_shape):
            raise ValueError("Cannot specify alpha for the axis #{}. "
                             "Maximum available axis is #{}"
                             "".format(max(alpha_axes), len(output_shape) - 1))

        alpha_shape = [output_shape[axis - 1] for axis in alpha_axes]

        if isinstance(alpha, Initializer):
            alpha = alpha.sample(alpha_shape)

        self.alpha = create_shared_parameter(
            value=alpha,
            name='alpha_{}'.format(self.layer_id),
            shape=alpha_shape
        )
        self.parameters.append(self.alpha)

    def activation_function(self, input_value):
        alpha = dimshuffle(self.alpha, input_value.ndim, self.alpha_axes)
        return T.nnet.relu(input_value, alpha)
