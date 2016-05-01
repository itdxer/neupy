from __future__ import division

import math

import six
import theano
import theano.tensor as T
import numpy as np

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

    # TODO: The None parameters that get useful only in
    # case of dill.load operation don't look good.
    # I should find a better way to solve this problem
    # The same I need to do with `__init__` method
    def __new__(cls, connection=None, options=None, floatX=None, **kwargs):
        # Argument `options` is a simple hack for the `__reduce__` method.
        # `__reduce__` can't retore class with keyword arguments and
        # it will put them as `dict` argument in the `options` and method
        # will translate it to kwargs. The same hack is in the
        # `__init__` method.

        if options is None:
            options = kwargs

        if floatX is not None:
            theano.config.floatX = floatX

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

    def __init__(self, connection, options=None, floatX=None, **kwargs):
        if options is None:
            options = kwargs
        super(GradientDescent, self).__init__(connection, **options)

    def init_param_updates(self, layer, parameter):
        step = self.variables.step
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
        floatX = theano.config.floatX
        args = (self.connection, parameters, floatX)
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
    """ Iterates batch slices.

    Parameters
    ----------
    n_samples : int
        Number of samples. Number should be greater than 0.
    batch_size : int
        Batch size. Number should be greater than 0.

    Yields
    ------
    object
        Batch slices.
    """
    n_batches = int(math.ceil(n_samples / batch_size))

    for batch_index in range(n_batches):
        yield slice(
            batch_index * batch_size,
            (batch_index + 1) * batch_size
        )


def cannot_divide_into_batches(data, batch_size):
    """ Checkes whether data can be divided into at least
    two batches.

    Parameters
    ----------
    data : array-like
    batch_size : int or None

    Returns
    -------
    bool
    """
    n_samples = len(data)
    return batch_size is None or n_samples <= batch_size


def apply_batches(function, arguments, batch_size, logger, description='',
                  show_progressbar=False, use_error_output=True):
    """ Apply batches to a specified function.

    Parameters
    ----------
    function : func
        Function that accepts one or more positional arguments.
        Each of them should be an array-like variable that
        have exactly the same number of rows.
    arguments : tuple, list
        The arguemnts that will be provided to the function specified
        in the ``function`` argument.
    batch_size : int
        Batch size.
    logger : TerminalLogger instance
    description : str
        Short description that will be displayed near the progressbar
        in verbose mode. Defaults to ``''`` (empty string).
    show_progressbar : bool
        ``True`` mean that function will show progressbar in the
        terminal. Defaults to ``False``.

    Returns
    -------
    list
        List of function outputs.
    """
    if not arguments:
        raise ValueError("The arguments parameter should have at "
                         "least one element.")

    samples = arguments[0]
    n_samples = len(samples)
    batch_iterator = iter_batches(n_samples, batch_size)

    if show_progressbar:
        batch_iterator = logger.progressbar(
            list(batch_iterator),
            desc=description,
            file=logger.stdout
        )

    output = None
    outputs = []
    for batch in batch_iterator:
        if show_progressbar and logger.enable:
            batch = batch_iterator.send(output if use_error_output else None)

        sliced_arguments = [argument[batch] for argument in arguments]
        output = function(*sliced_arguments)

        outputs.append(output)

    return outputs


def average_batch_errors(errors, n_samples, batch_size):
    """ Computes average error per sample.

    Parameters
    ----------
    errors : list
        List of errors where each element is a average error
        per batch.
    n_samples : int
        Number of samples in the dataset.
    batch_size : int
        Batch size.

    Returns
    -------
    float
        Average error per sample.
    """
    n_samples_in_final_batch = n_samples % batch_size

    if n_samples_in_final_batch == 0:
        return np.mean(errors)

    all_errors_without_last = errors[:-1]
    last_error = errors[-1]

    total_error = (
        sum(all_errors_without_last) * batch_size +
        last_error * n_samples_in_final_batch
    )
    average_error = (total_error / n_samples)
    return average_error


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

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms
    >>>
    >>> x_train = np.array([[1, 2], [3, 4]])
    >>> y_train = np.array([[1], [0]])
    >>>
    >>> mgdnet = algorithms.MinibatchGradientDescent(
    ...     (2, 3, 1), batch_size=1
    ... )
    >>> mgdnet.train(x_train, y_train)

    See Also
    --------
    :network:`GradientDescent` : GradientDescent algorithm.
    """
    batch_size = BatchSizeProperty(default=100)

    def train_epoch(self, input_train, target_train):
        """ Train one epoch.

        Parameters
        ----------
        input_train : array-like
            Training input array.
        target_train : array-like
            Training target array.

        Returns
        -------
        float
            Training error.
        """
        train_epoch = self.methods.train_epoch

        if cannot_divide_into_batches(input_train, self.batch_size):
            return train_epoch(input_train, target_train)

        show_progressbar = (self.training and self.training.show_epoch == 1)
        errors = apply_batches(
            function=train_epoch,
            arguments=(input_train, target_train),
            batch_size=self.batch_size,

            description='Training batches',
            show_progressbar=show_progressbar,
            logger=self.logs,
            use_error_output=True,
        )
        return average_batch_errors(
            errors,
            n_samples=len(input_train),
            batch_size=self.batch_size,
        )

    def prediction_error(self, input_data, target_data):
        """ Check the prediction error for the specified input samples
        and their targets.

        Parameters
        ----------
        input_data : array-like
        target_data : array-like

        Returns
        -------
        float
            Prediction error.
        """
        input_data = self.format_input_data(input_data)
        target_data = self.format_target_data(target_data)

        prediction_error = self.methods.prediction_error

        if cannot_divide_into_batches(input_data, self.batch_size):
            return prediction_error(input_data, target_data)

        show_progressbar = (self.training and self.training.show_epoch == 1)
        errors = apply_batches(
            function=prediction_error,
            arguments=(input_data, target_data),
            batch_size=self.batch_size,

            description='Validation batches',
            show_progressbar=show_progressbar,
            logger=self.logs,
            use_error_output=True,
        )
        return average_batch_errors(
            errors,
            n_samples=len(input_data),
            batch_size=self.batch_size,
        )

    def predict_raw(self, input_data):
        """ Makes a raw prediction.

        Parameters
        ----------
        input_data : array-like

        Returns
        -------
        array-like
        """
        input_data = self.format_input_data(input_data)
        predict_raw = self.methods.predict_raw

        if cannot_divide_into_batches(input_data, self.batch_size):
            return predict_raw(input_data)

        outputs = apply_batches(
            function=predict_raw,
            arguments=(input_data,),
            batch_size=self.batch_size,

            description='Prediction batches',
            show_progressbar=True,
            logger=self.logs,
            use_error_output=False,
        )

        return np.concatenate(outputs, axis=0)
