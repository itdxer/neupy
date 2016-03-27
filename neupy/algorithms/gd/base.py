from __future__ import division

import math

import six
import theano.tensor as T

from neupy.core.properties import Property, BoundedProperty
from neupy.network import ConstructableNetwork
from . import addon_types


__all__ = ('GradientDescent', 'MinibatchGradientDescent')


class GradientDescent(ConstructableNetwork):
    """ GradientDescent algorithm.

    Parameters
    ----------
    addons : list or None
        The list of addon algortihms. ``None`` by default.
        If this option is not empty it will generate new class which
        will inherit all from this list. Support two types of
        addon algorithms: weight update and step update.
    {ConstructableNetwork.connection}
    {ConstructableNetwork.error}
    {BaseNetwork.step}
    {BaseNetwork.show_epoch}
    {BaseNetwork.shuffle_data}
    {BaseNetwork.epoch_end_signal}
    {BaseNetwork.train_end_signal}
    {Verbose.verbose}

    Attributes
    ----------
    {BaseNetwork.errors}
    {BaseNetwork.train_errors}
    {BaseNetwork.validation_errors}
    {BaseNetwork.last_epoch}

    Methods
    -------
    {BaseSkeleton.predict}
    {SupervisedLearning.train}
    {BaseSkeleton.fit}
    {BaseNetwork.plot_errors}

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
    supported_addon_types = addon_types.keys()

    addons = Property(default=None, expected_type=list)

    def __new__(cls, connection=None, options=None, **kwargs):
        # Argument `options` is a simple hack for the `__reduce__` method.
        # `__reduce__` can't retore class with keyword arguments and
        # it will put them as `dict` argument in the `options` and method
        # will translate it to kwargs. The same hack is in the
        # `__init__` method.
        if options is None:
            options = kwargs

        addons = options.get('addons')

        if not addons:
            cls.main_class = cls
            return super(GradientDescent, cls).__new__(cls)

        identified_types = []
        for addon_class in addons:
            opt_class_type = getattr(addon_class, 'addon_type',  None)

            if opt_class_type not in cls.supported_addon_types:
                opt_class_name = addon_class.__name__
                supported_opts = ', '.join(addon_types.values())
                raise ValueError(
                    "Invalid add-on class `{}`. Class supports only "
                    "{}".format(opt_class_name, supported_opts)
                )

            if opt_class_type in identified_types:
                raise ValueError(
                    "There can be only one add-on class with "
                    "type `{}`".format(addon_types[opt_class_type])
                )

            identified_types.append(opt_class_type)

        new_class_name = (
            cls.__name__ +
            ''.join(class_.__name__ for class_ in addons)
        )
        mro_classes = tuple(list(addons) + [cls])
        new_class = type(new_class_name, mro_classes, {})
        new_class.main_class = cls

        return super(GradientDescent, new_class).__new__(new_class)

    def __init__(self, connection, options=None, **kwargs):
        if options is None:
            options = kwargs
        super(GradientDescent, self).__init__(connection, **options)

    def init_param_updates(self, layer, parameter):
        step = layer.step or self.variables.step
        gradient = T.grad(self.variables.error_func, wrt=parameter)
        return [(parameter, parameter - step * gradient)]

    def class_name(self):
        return self.main_class.__name__

    def get_params(self, deep=False, with_connection=True):
        params = super(GradientDescent, self).get_params()
        if with_connection:
            params['connection'] = self.connection
        return params

    def __reduce__(self):
        parameters = self.get_params(with_connection=False)
        args = (self.connection, parameters)
        return (self.main_class, args)


class BatchSizeProperty(BoundedProperty):
    """ Batch size property

    Parameters
    ----------
    {BoundedProperty.maxval}
    {BaseProperty.default}
    {BaseProperty.required}
    """
    expected_type = (type(None), int)
    fullbatch_identifiers = [None, -1, 'all', '*', 'full']

    def __init__(self, *args, **kwargs):
        super(BatchSizeProperty, self).__init__(minval=1, *args, **kwargs)

    def __set__(self, instance, value):
        if isinstance(value, six.string_types):
            value = value.lower()

        if value in self.fullbatch_identifiers:
            value = None

        super(BatchSizeProperty, self).__set__(instance, value)

    def validate(self, value):
        if value is not None:
            super(BatchSizeProperty, self).validate(value)


def iter_batches(n_samples, batch_size):
    n_batches = int(math.floor(n_samples / batch_size))

    for batch_index in range(n_batches):
        yield slice(
            batch_index * batch_size,
            (batch_index + 1) * batch_size
        )


class MinibatchGradientDescent(GradientDescent):
    """ Mini-batch Gradient Descent algorithm.

    Parameters
    ----------
    batch_size : int or {{None, -1, 'all', '*', 'full'}}
        Set up batch size for learning process. To set up batch size equal to
        sample size value should be equal to one of the values listed above.
        Defaults to ``100``.
    {GradientDescent.addons}
    {ConstructableNetwork.connection}
    {ConstructableNetwork.error}
    {BaseNetwork.step}
    {BaseNetwork.show_epoch}
    {BaseNetwork.shuffle_data}
    {BaseNetwork.epoch_end_signal}
    {BaseNetwork.train_end_signal}
    {Verbose.verbose}

    Attributes
    ----------
    {BaseNetwork.errors}
    {BaseNetwork.train_errors}
    {BaseNetwork.validation_errors}
    {BaseNetwork.last_epoch}

    Methods
    -------
    {BaseSkeleton.predict}
    {SupervisedLearning.train}
    {BaseSkeleton.fit}
    {BaseNetwork.plot_errors}

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
        """ Network training epoch.

        Parameters
        ----------
        input_train : array-like
            Training input array.
        target_train : array-like
            Training target array.

        Returns
        -------
        float
            Train data prediction error based on chosen
            error function.
        """
        n_samples = len(input_train)
        batch_size = self.batch_size
        train_epoch = self.methods.train_epoch

        if batch_size is None or n_samples <= batch_size:
            return train_epoch(input_train, target_train)

        batch_iterator = iter_batches(n_samples, batch_size)

        if self.training and self.training.show_epoch == 1:
            batch_iterator = self.logs.progressbar(
                list(batch_iterator),
                desc='Train batches',
                file=self.logs.stdout
            )

        total_error = 0
        for batch in batch_iterator:
            total_error += train_epoch(input_train[batch],
                                       target_train[batch])

        average_error = batch_size * total_error / n_samples
        return average_error

    def prediction_error(self, input_data, target_data):
        input_data = self.format_input_data(input_data)
        target_data = self.format_target_data(target_data)

        n_samples = len(input_data)
        batch_size = self.batch_size
        prediction_error = self.methods.prediction_error

        if batch_size is None or n_samples <= batch_size:
            return prediction_error(input_data, target_data)

        batch_iterator = iter_batches(n_samples, batch_size)

        if self.training and self.training.show_epoch == 1:
            batch_iterator = self.logs.progressbar(
                list(batch_iterator),
                desc='Validation batches',
                file=self.logs.stdout
            )

        total_error = 0
        for batch in batch_iterator:
            total_error += prediction_error(input_data[batch],
                                            target_data[batch])

        average_error = batch_size * total_error / n_samples
        return average_error
