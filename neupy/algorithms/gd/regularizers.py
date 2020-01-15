from __future__ import division

import re
from collections import defaultdict

import six
import tensorflow as tf

from neupy.utils import asfloat, as_tuple


__all__ = ('Regularizer', 'l1', 'l2', 'maxnorm')


def does_layer_match_pattern(layer, layer_identifier):
    if layer_identifier is Ellipsis:
        return True
    if isinstance(layer_identifier, six.string_types):
        return bool(re.match(layer_identifier, layer.name))
    return isinstance(layer, layer_identifier)


def does_parameter_match_pattern(parameter_name, pattern):
    if pattern is Ellipsis:
        return True
    return bool(re.match(pattern, parameter_name))


def layer_and_parameter_matched(layer, parameter_name, patterns):
    if not patterns:
        return False

    for layer_identifier, parameter_pattern in patterns.items():
        layer_matched = does_layer_match_pattern(layer, layer_identifier)
        pattern_matched = does_parameter_match_pattern(parameter_name, parameter_pattern)

        if layer_matched and pattern_matched:
            return True

    return False


class Regularizer(object):
    def __init__(self, function, include=None, exclude=None, verbose=False, *args, **kwargs):
        self.function = function

        self.include = include
        self.exclude = exclude
        self.verbose = verbose

        self.args = args
        self.kwargs = kwargs

    def __call__(self, network):
        cost = asfloat(0)
        statistics = {"included": defaultdict(int), "excluded": defaultdict(int)}

        for (layer, parameter_name), variable in network.variables.items():
            if not variable.trainable:
                continue

            parameter_could_be_included = layer_and_parameter_matched(layer, parameter_name, self.include)
            parameter_could_be_excluded = layer_and_parameter_matched(layer, parameter_name, self.exclude)
            layer_class_name = layer.__class__.__name__

            if parameter_could_be_included and not parameter_could_be_excluded:
                cost += self.function(variable, *self.args, **self.kwargs)
                statistics["included"][(layer_class_name, parameter_name)] += 1
            else:
                statistics["excluded"][(layer_class_name, parameter_name)] += 1

        if self.verbose:
            self.print_statistics(statistics)

        return cost

    def print_statistics(self, statistics):
        print("Statistics from the {} regularizer".format(self.__class__.__name__))
        print("  Included trainable parameters:")
        for (layer_class_name, parameter_name), count in sorted(statistics["included"].items()):
            print("    layer: {}, parameter: {}, count: {}".format(layer_class_name, parameter_name, count))

        print("  Excluded trainable parameters:")
        for (layer_class_name, parameter_name), count in sorted(statistics["excluded"].items()):
            print("    layer: {}, parameter: {}, count: {}".format(layer_class_name, parameter_name, count))

    def __repr__(self):
        kwargs_repr = [repr(arg) for arg in self.args]
        kwargs_repr += ["{}={}".format(k, v) for k, v in self.kwargs.items()]
        kwargs_repr += ["exclude={}".format(self.exclude)]
        return "{}({})".format(self.__class__.__name__, ', '.join(kwargs_repr))


class l1(Regularizer):
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
    def __init__(self, decay_rate=0.01, *args, **kwargs):
        self.decay_rate = decay_rate
        super(l2, self).__init__(function=self.l1_loss, *args, **kwargs)

    def l1_loss(self, weight):
        return tf.multiply(self.decay_rate, tf.reduce_sum(tf.abs(weight)))


class l2(Regularizer):
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
    def __init__(self, decay_rate=0.01, *args, **kwargs):
        self.decay_rate = decay_rate
        super(l2, self).__init__(function=self.l2_loss, *args, **kwargs)

    def l2_loss(self, weight):
        return tf.multiply(self.decay_rate, tf.reduce_sum(tf.pow(weight, 2)))


class maxnorm(Regularizer):
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
    def __init__(self, decay_rate=0.01, *args, **kwargs):
        self.decay_rate = decay_rate
        super(l2, self).__init__(function=self.maxnorm_loss, *args, **kwargs)

    def maxnorm_loss(self, weight):
        return tf.multiply(self.decay_rate, tf.reduce_max(tf.abs(weight)))
