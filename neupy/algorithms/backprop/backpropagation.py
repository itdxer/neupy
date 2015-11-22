from itertools import chain

import theano
import theano.tensor as T

from neupy.utils import asfloat, AttributeKeyDict
from neupy.core.properties import ListProperty, ChoiceProperty
from neupy.network.learning import SupervisedLearning
from neupy.network.base import ConstructableNetwork
from neupy.network.errors import (mse, binary_crossentropy,
                                  categorical_crossentropy)
from . import optimization_types


__all__ = ('Backpropagation',)


class Backpropagation(SupervisedLearning, ConstructableNetwork):
    """ Backpropagation algorithm.

    Parameters
    ----------
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

        self.variables = AttributeKeyDict()
        self.init_variables()
        self.init_methods()

    def init_variables(self):
        """ Initialize Theano variables.
        """

        network_input = T.matrix('x')
        network_output = T.matrix('y')

        layer_input = network_input
        for layer in self.train_layers:
            layer_input = layer.output(layer_input)
        prediction = layer_input

        self.variables.update(
            network_input=network_input,
            network_output=network_output,
            step=theano.shared(name='step', value=asfloat(self.step)),
            epoch=theano.shared(name='epoch', value=1, borrow=False),
            error_func=self.error(network_output, prediction),
            prediction_func=prediction,
        )

    def init_methods(self):
        """ Initialize all methods that needed for prediction and
        training procedures.
        """

        network_input = self.variables.network_input
        network_output = self.variables.network_output

        self.train_epoch = theano.function(
            inputs=[network_input, network_output],
            outputs=self.variables.error_func,
            updates=self.init_train_updates(),
        )
        self.prediction_error = theano.function(
            inputs=[network_input, network_output],
            outputs=self.variables.error_func
        )
        self.predict_raw = theano.function(
            inputs=[network_input],
            outputs=self.variables.prediction_func
        )

    def init_train_updates(self):
        """ Initialize train function update in Theano format that
        would be trigger after each trainig epoch.
        """
        updates = []
        for layer in self.train_layers:
            updates.extend(self.init_layer_updates(layer))
        return updates

    def init_layer_updates(self, layer):
        updates = []
        for parameter in layer.parameters:
            updates.extend(self.init_param_updates(layer, parameter))
        return updates

    def init_param_updates(self, layer, parameter):
        step = layer.step or self.variables.step
        gradient = T.grad(self.variables.error_func, wrt=parameter)
        return [(parameter, parameter - step * gradient)]

    def epoch_start_update(self, epoch):
        super(Backpropagation, self).epoch_start_update(epoch)
        self.variables.epoch.set_value(epoch)

    def class_name(self):
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
