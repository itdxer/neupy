import math
from itertools import chain

import six
import theano.tensor as T

from neupy.core.properties import (ListProperty, ChoiceProperty, IntProperty,
                                   BoundedProperty)
from neupy.network import SupervisedConstructableNetwork
from neupy.network.errors import (mse, binary_crossentropy,
                                  categorical_crossentropy)
from . import optimization_types


__all__ = ('GradientDescent', 'MinibatchGradientDescent')


class GradientDescent(SupervisedConstructableNetwork):
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


class BatchSizeProperty(IntProperty, BoundedProperty):
    expected_type = (type(None), int)
    fullbatch_identifiers = [None, -1, 'all', '*', 'full']
    min_size = 1

    def __set__(self, instance, value):
        if isinstance(value, six.string_types):
            value = value.lower()

        if value in self.fullbatch_identifiers:
            value = None

        super(BatchSizeProperty, self).__set__(instance, value)

    def validate(self, value):
        if value is not None:
            super(BatchSizeProperty, self).validate(value)


class MinibatchGradientDescent(GradientDescent):
    """ Mini-batch Gradient Descent algorithm.

    Parameters
    ----------
    batch_size : int or {{None, -1, 'all', '*', 'full'}}
        Set up batch size for learning process. To set up batch size equal to
        sample size value should be equal to one of the values listed above.
        Defaults to ``100``.
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
    batch_size = BatchSizeProperty(default=100)

    def train_epoch(self, input_train, target_train):
        """ Network training.
        """

        n_samples = len(input_train)
        batch_size = self.batch_size
        train_epoch = self.methods.train_epoch

        if batch_size is None:
            return train_epoch(input_train, target_train)

        n_batches = math.floor(n_samples / batch_size)
        prediction_error = self.methods.prediction_error

        for batch_index in range(n_batches):
            slice_batch = slice(batch_index * batch_size,
                                (batch_index + 1) * batch_size)
            train_epoch(input_train[slice_batch], target_train[slice_batch])

        return prediction_error(input_train, target_train)
