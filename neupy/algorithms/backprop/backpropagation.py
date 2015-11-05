from itertools import chain

import theano.tensor as T

from neupy.core.properties import ListProperty
from neupy.network.learning import SupervisedLearning
from neupy.network.base import BaseNetwork
from . import optimization_types


__all__ = ('Backpropagation',)


class Backpropagation(SupervisedLearning, BaseNetwork):
    """ Backpropagation algorithm.

    Parameters
    ----------
    {optimizations}
    {full_params}

    Methods
    -------
    {supervised_train}
    {raw_predict}
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
        new_class.main_class = cls

        return super(Backpropagation, new_class).__new__(new_class)

    def __init__(self, connection, options=None, **kwargs):
        if options is None:
            options = kwargs

        self.optimizations = default_optimizations = self.default_optimizations

        if default_optimizations and 'optimizations' in options:
            optimizations_merged = []
            optimizations = options['optimizations']

            for algorithm in chain(optimizations, default_optimizations):
                types = [alg.optimization_type for alg in optimizations_merged]

                if algorithm.optimization_type not in types:
                    optimizations_merged.append(algorithm)

            options['optimizations'] = optimizations_merged

        super(Backpropagation, self).__init__(connection, **options)

    def init_train_updates(self):
        updates = [(self.variables.epoch, self.variables.epoch + 1)]
        step = self.variables.step

        for layer in self.train_layers:
            grad_w = T.grad(self.cost, wrt=layer.weight)
            updates.append((layer.weight, layer.weight - step * grad_w))

            if layer.use_bias:
                grad_b = T.grad(self.cost, wrt=layer.bias)
                updates.append((layer.bias, layer.bias - step * grad_b))

        return updates

    def get_class_name(self):
        return 'Backpropagation'

    def get_params(self, deep=False, with_connection=True):
        params = super(Backpropagation, self).get_params()
        if with_connection:
            params['connection'] = self.connection
        return params

    def __reduce__(self):
        args = (self.connection, self.get_params(with_connection=False))

        # Main class should be different if we construct it dynamicaly
        if hasattr(self, 'main_class'):
            main_class = self.main_class
        else:
            main_class = self.__class__

        return (main_class, args)
