from __future__ import division

from functools import wraps

import tensorflow as tf

from neupy.utils import asfloat


__all__ = ('define_regularizer', 'l1', 'l2', 'maxnorm')


class Regularizer(object):
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.exclude = kwargs.pop('exclude', ['bias'])

        self.args = args
        self.kwargs = kwargs

    def __call__(self, network):
        cost = asfloat(0)

        for (layer, varname), variable in network.variables.items():
            if varname not in self.exclude and variable.trainable:
                cost += self.function(variable, *self.args, **self.kwargs)

        return cost

    def __repr__(self):
        kwargs_repr = [repr(arg) for arg in self.args]
        kwargs_repr += ["{}={}".format(k, v) for k, v in self.kwargs.items()]
        kwargs_repr += ["exclude={}".format(self.exclude)]
        return "{}({})".format(self.function.__name__, ', '.join(kwargs_repr))


def define_regularizer(function):
    """
    Wraps regularization function and uses it in order to apply
    regularization per each parameter in the network.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from neupy import algorithms
    >>> from neupy.layers import *
    >>>
    >>> @algorithms.define_regularizer
    ... def l2(weight, decay_rate=0.01):
    ...     return decay_rate * tf.reduce_sum(tf.pow(weight, 2))
    ...
    >>> l2_regularizer = l2(dacay_rate=0.01)
    >>> network = Input(5) >> Relu(10) >> Sigmoid(1)
    >>> reguarization_cost = l2_regularizer(network)
    """
    @wraps(function)
    def wrapper(*args, **kwargs):
        return Regularizer(function, *args, **kwargs)
    return wrapper


@define_regularizer
def l1(weight, decay_rate=0.01):
    """
    Applies l1 regularization to the trainable parameters in the network.

    Regularization cost per weight parameter in the layer can be computed
    in the following way (pseudocode).

    .. code-block:: python

        cost = decay_rate * sum(abs(weight))

    Parameters
    ----------
    decay_rate : float
        Controls training penalties during the parameter updates.
        The larger the value the stronger effect regularization
        has during the training. Defaults to ``0.01``.

    exclude : list
        List of parameter names that has to be excluded from the
        regularization. Defaults to ``['bias']``.

    Examples
    --------
    >>> from neupy import algorithms
    >>> from neupy.layers import *
    >>>
    >>> optimizer = algorithms.Momentum(
    ...     Input(5) >> Relu(10) >> Sigmoid(1),
    ...     step=algorithms.l1(decay_rate=0.01)
    ... )

    With included regularization for bias

    >>> from neupy import algorithms
    >>> from neupy.layers import *
    >>>
    >>> optimizer = algorithms.Momentum(
    ...     Input(5) >> Relu(10) >> Sigmoid(1),
    ...     step=algorithms.l1(decay_rate=0.01, exclude=[])
    ... )
    """
    return tf.multiply(decay_rate, tf.reduce_sum(tf.abs(weight)))


@define_regularizer
def l2(weight, decay_rate=0.01):
    """
    Applies l2 regularization to the trainable parameters in the network.

    Regularization cost per weight parameter in the layer can be computed
    in the following way (pseudocode).

    .. code-block:: python

        cost = decay_rate * sum(weight ** 2)

    Parameters
    ----------
    decay_rate : float
        Controls training penalties during the parameter updates.
        The larger the value the stronger effect regularization
        has during the training. Defaults to ``0.01``.

    exclude : list
        List of parameter names that has to be excluded from the
        regularization. Defaults to ``['bias']``.

    Examples
    --------
    >>> from neupy import algorithms
    >>> from neupy.layers import *
    >>>
    >>> optimizer = algorithms.Momentum(
    ...     Input(5) >> Relu(10) >> Sigmoid(1),
    ...     step=algorithms.l2(decay_rate=0.01)
    ... )

    With included regularization for bias

    >>> from neupy import algorithms
    >>> from neupy.layers import *
    >>>
    >>> optimizer = algorithms.Momentum(
    ...     Input(5) >> Relu(10) >> Sigmoid(1),
    ...     step=algorithms.l2(decay_rate=0.01, exclude=[])
    ... )
    """
    return tf.multiply(decay_rate, tf.reduce_sum(tf.pow(weight, 2)))


@define_regularizer
def maxnorm(weight, decay_rate=0.01):
    """
    Applies max-norm regularization to the trainable parameters in the
    network. Also known and l-inf regularization.

    Regularization cost per weight parameter in the layer can be computed
    in the following way (pseudocode).

    .. code-block:: python

        cost = decay_rate * max(abs(weight))

    Parameters
    ----------
    decay_rate : float
        Controls training penalties during the parameter updates.
        The larger the value the stronger effect regularization
        has during the training. Defaults to ``0.01``.

    exclude : list
        List of parameter names that has to be excluded from the
        regularization. Defaults to ``['bias']``.

    Examples
    --------
    >>> from neupy import algorithms
    >>> from neupy.layers import *
    >>>
    >>> optimizer = algorithms.Momentum(
    ...     Input(5) >> Relu(10) >> Sigmoid(1),
    ...     step=algorithms.maxnorm(decay_rate=0.01)
    ... )

    With included regularization for bias

    >>> from neupy import algorithms
    >>> from neupy.layers import *
    >>>
    >>> optimizer = algorithms.Momentum(
    ...     Input(5) >> Relu(10) >> Sigmoid(1),
    ...     step=algorithms.maxnorm(decay_rate=0.01, exclude=[])
    ... )
    """
    return tf.multiply(decay_rate, tf.reduce_max(tf.abs(weight)))
