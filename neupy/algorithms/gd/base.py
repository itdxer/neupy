import math
from itertools import chain

import theano
import theano.tensor as T

from neupy.core.properties import ListProperty, ChoiceProperty, NumberProperty
from neupy.network.learning import SupervisedLearning
from neupy.network.base import ConstructableNetwork
from neupy.network.errors import (mse, binary_crossentropy,
                                  categorical_crossentropy)
from . import optimization_types


__all__ = ('GradientDescent', 'MinibatchGradientDescent')


class GradientDescent(SupervisedLearning, ConstructableNetwork):
    """ GradientDescent algorithm.

    Parameters
    ----------
    {connection}
    {optimizations}
    {full_params}

    Methods
    -------
    {supervised_train}
    {predict_raw}
    {full_methods}

    Examples
    --------
    >>> import numpy as np
    >>> from neupy.algorithms import GradientDescent
    >>>
    >>> x_train = np.array([[1, 2], [3, 4]])
    >>> y_train = np.array([[1], [0]])
    >>>
    >>> bpnet = GradientDescent((2, 3, 1), verbose=False, step=0.1)
    >>> bpnet.train(x_train, y_train)
    """

    __opt_params = """optimizations : list or None
        The list of optimization algortihms. ``None`` by default.
        If this option is not empty it will generate new class which
        will inherit all from this list. Support two types of
        optimization algorithms: weight update and step update.
    """
    shared_docs = {"optimizations": __opt_params}
    default_optimizations = []

    error = ChoiceProperty(default='mse', choices={
        'mse': mse,
        'binary_crossentropy': binary_crossentropy,
        'categorical_crossentropy': categorical_crossentropy,
    })
    optimizations = ListProperty(default=None)

    def __new__(cls, connection, options=None, **kwargs):
        # Argument `options` is a simple hack for the `__reduce__` method.
        # `__reduce__` can't retore class with keyword arguments and
        # it will put them as `dict` argument in the `options` and method
        # will translate it to kwargs. The same hack is in the
        # `__init__` method.
        if options is None:
            options = kwargs

        optimizations = options.get('optimizations', cls.default_optimizations)
        if not optimizations:
            return super(GradientDescent, cls).__new__(cls)

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

        return super(GradientDescent, new_class).__new__(new_class)

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

        super(GradientDescent, self).__init__(connection, **options)

    def init_param_updates(self, layer, parameter):
        step = layer.step or self.variables.step
        gradient = T.grad(self.variables.error_func, wrt=parameter)
        return [(parameter, parameter - step * gradient)]

    def class_name(self):
        return 'GradientDescent'

    def get_params(self, deep=False, with_connection=True):
        params = super(GradientDescent, self).get_params()
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


class MinibatchGradientDescent(GradientDescent):
    """ Mini-batch Gradient Descent algorithm.

    Parameters
    ----------
    batch_size : int
        Setup batch size for learning process. Defaults to ``10``.
    {optimizations}
    {full_params}

    Methods
    -------
    {supervised_train}
    {predict_raw}
    {full_methods}

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms
    >>>
    >>> x_train = np.array([[1, 2], [3, 4]])
    >>> y_train = np.array([[1], [0]])
    >>>
    >>> mgdnet = algorithms.MinibatchGradientDescent(
    ...     (2, 3, 1),
    ...     verbose=False,
    ...     batch_size=1
    ... )
    >>> mgdnet.train(x_train, y_train)

    See Also
    --------
    :network:`GradientDescent` : GradientDescent algorithm.
    """
    batch_size = NumberProperty(default=100)

    def init_variables(self):
        super(MinibatchGradientDescent, self).init_variables()
        self.variables.update(batch_index=T.lscalar())

    def train_epoch(self, input_train, target_train):
        """ Network training.
        """

        n_samples = len(input_train)
        n_batches = math.floor(n_samples / self.batch_size)

        network_input = self.variables.network_input
        network_output = self.variables.network_output
        batch_index = self.variables.batch_index
        prediction_error = self.methods.prediction_error

        slice_batch = slice(batch_index * (self.batch_size),
                            (batch_index + 1) * (self.batch_size))

        input_train_shared = theano.shared(name="x_train", value=input_train)
        target_train_shared = theano.shared(name="y_train", value=target_train)

        train_batch = theano.function(
            inputs=[batch_index],
            outputs=self.variables.error_func,
            updates=self.init_train_updates(),
            givens={
                network_input: input_train_shared[slice_batch],
                network_output: target_train_shared[slice_batch],
            }
        )

        for batch_index in range(n_batches):
            batch_error = train_batch(batch_index)

        return prediction_error(input_train, target_train)
