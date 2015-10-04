from itertools import chain

from numpy import dot, multiply

from neupy.core.properties import ListProperty
from neupy.algorithms.feedforward import FeedForwardNetwork
from neupy.network.learning import SupervisedLearning
from . import optimization_types


__all__ = ('Backpropagation',)


class Backpropagation(SupervisedLearning, FeedForwardNetwork):
    """ Backpropagation algorithm.

    Parameters
    ----------
    {optimizations}
    {raw_predict_param}
    {full_params}

    Methods
    -------
    {supervised_train}
    {full_methods}

    Examples
    --------
    >>> import numpy as np
    >>> from neupy.algorithms import Backpropagation
    >>>
    >>> x_train = np.array([[1, 2], [3, 4]])
    >>> y_train = np.array([[1], [0]])
    >>>
    >>> bpnet = Backpropagation((2, 3, 1), verbose=False, step=0.1)
    >>> bpnet.train(x_train, y_train)
    """

    __opt_params = """optimizations : list or None
        The list of optimization algortihms. ``None`` by default.
        If this option is not empty it will generate new class which
        will inherit all from this list. Support two types of
        optimization algorithms: weight update and step update.
    """
    shared_docs = {"optimizations": __opt_params}

    optimizations = ListProperty(default=None)
    default_optimizations = []

    def __new__(cls, connection, options=None, **kwargs):
        # Argument `options` is a simple hack for the `__reduce__` method.
        # `__reduce__` can't retore class with keyword arguments and
        # it will put them as `dict` argument in the `options` and method
        # will translate it to kwargs. The same hack is at the
        # `__init__` method.
        if options is None:
            options = kwargs

        optimizations = options.get('optimizations', cls.default_optimizations)
        if not optimizations:
            return super(Backpropagation, cls).__new__(cls)

        founded_types = []
        for optimization_class in optimizations:
            opt_class_type = getattr(optimization_class, 'optimization_type',
                                     None)
            if opt_class_type not in optimization_types:
                raise ValueError("Invalid optimization class `{}`".format(
                                 optimization_class.__name__))

            if opt_class_type in founded_types:
                raise ValueError(
                    "There can be only one optimization class with "
                    "type `{}`".format(optimization_types[opt_class_type])
                )

            founded_types.append(opt_class_type)

        # Build new class which would inherit main and all optimization
        new_class_name = (
            cls.__name__ +
            ''.join(class_.__name__ for class_ in optimizations)
        )
        mro_classes = tuple(list(optimizations) + [cls])
        new_class = type(new_class_name, mro_classes, {})

        return super(Backpropagation, new_class).__new__(new_class)

    def __init__(self, connection, options=None, **kwargs):
        if options is None:
            options = kwargs

        default_optimizations = self.default_optimizations
        optimizations = options.get('optimizations', default_optimizations)

        if optimizations != default_optimizations:
            optimizations_merged = []
            for algorithm in chain(optimizations, default_optimizations):
                types = [alg.optimization_type for alg in optimizations_merged]
                if algorithm.optimization_type not in types:
                    optimizations_merged.append(algorithm)
        else:
            optimizations_merged = optimizations

        options['optimizations'] = optimizations_merged
        super(Backpropagation, self).__init__(connection, **options)

    def get_gradient(self, output_train, target_train):
        self.delta = []
        self.gradients = []

        update = self.error.deriv(output_train, target_train)

        for i, layer in enumerate(reversed(self.train_layers), start=1):
            summated_data = self.summated_data[-i]
            current_layer_input = self.layer_outputs[-i]

            deriv = layer.activation_function.deriv(summated_data)

            delta = multiply(deriv, update)
            update = dot(delta, layer.weight_without_bias.T)

            gradient = dot(current_layer_input.T, delta)

            self.gradients.insert(0, gradient)
            self.delta.insert(0, delta)

        return self.gradients

    def get_weight_delta(self, output_train, target_train):
        gradients = self.get_gradient(output_train, target_train)
        return [-gradient for gradient in gradients]

    def update_weights(self, weight_deltas):
        layer_weight_update = self.layer_weight_update
        for i, layer in enumerate(self.train_layers):
            layer.weight += layer_weight_update(weight_deltas[i], i)

    def layer_step(self, layer_number):
        return self.step

    def layer_weight_update(self, delta, layer_number):
        return self.layer_step(layer_number) * delta

    def get_class_name(self):
        return 'Backpropagation'

    def get_params(self, deep=False, with_connection=True):
        params = super(Backpropagation, self).get_params()
        if with_connection:
            params['connection'] = self.connection
        return params

    def __reduce__(self):
        args = (self.connection, self.get_params(with_connection=False))
        return (Backpropagation, args)
