from __future__ import division

import math
import collections

import six
import theano
import theano.tensor as T
import numpy as np

from neupy.core.config import Configurable
from neupy.core.properties import Property, BoundedProperty
from neupy.utils import as_tuple
from neupy.algorithms.constructor import ConstructibleNetwork
from neupy.algorithms.gd import addon_types


__all__ = ('GradientDescent', 'MinibatchGradientDescent')


class GradientDescent(ConstructibleNetwork):
    """
    Gradient descent algorithm.

    Parameters
    ----------
    {ConstructibleNetwork.Parameters}

    addons : list or None
        The list of addon algortihms. ``None`` by default.
        If this option is not empty it will generate new class which
        will inherit all from this list. Support two types of
        addon algorithms: weight update and step update.

    Attributes
    ----------
    {ConstructibleNetwork.Attributes}

    Methods
    -------
    {ConstructibleNetwork.Methods}

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

    # TODO: The arguments that have default value equal to `None`
    # are useful only in case if we need to save network in the
    # file. This solution looks bad and I need to redesign it later.
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
                    "Invalid add-on class '{}'. Class supports only "
                    "{}".format(opt_class_name, supported_opts)
                )

            if opt_class_type in identified_types:
                raise ValueError(
                    "There can be only one add-on class with "
                    "type '{}'".format(addon_types[opt_class_type])
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
    """
    Batch size property

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
    """
    Iterates batch slices.

    Parameters
    ----------
    n_samples : int
        Number of samples. Number should be greater than ``0``.

    batch_size : int
        Mini-batch size. Number should be greater than ``0``.

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
    """
    Checkes whether data can be divided into at least
    two batches.

    Parameters
    ----------
    data : array-like
        Dataset.

    batch_size : int or None
        Size of the batch.

    Returns
    -------
    bool
    """
    if isinstance(data, (list, tuple)):
        # In case if network has more than one input
        data = data[0]

    n_samples = len(data)
    return batch_size is None or n_samples <= batch_size


def format_error(error):
    """
    Format the error value.

    Parameters
    ----------
    error : float, list or None

    Returns
    -------
    str
        Formated error value.

    Examples
    --------
    >>> format_error(None)
    '?'
    >>>
    >>> format_error(0.43)
    '0.43000'
    >>>
    >>> format_error(1 / 3.)
    '0.33333'
    """
    if error is None:
        return '?'

    if isinstance(error, collections.Iterable):
        error = np.atleast_1d(error).item(0)

    if abs(error) > 100:
        return "{:.0f}".format(error)

    return '{:.5f}'.format(error)


def apply_batches(function, arguments, batch_size, logger, description='',
                  show_progressbar=False, show_error_output=True):
    """
    Apply batches to a specified function.

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
        Mini-batch size.

    logger : TerminalLogger instance

    description : str
        Short description that will be displayed near the progressbar
        in verbose mode. Defaults to ``''`` (empty string).

    show_progressbar : bool
        ``True`` means that function will show progressbar in the
        terminal. Defaults to ``False``.

    show_error_output : bool
        Assumes that outputs from the function errors.
        ``True`` will show information in the progressbar.
        Error will be related to the last epoch.
        Defaults to ``True``.

    Returns
    -------
    list
        List of function outputs.
    """
    if not arguments:
        raise ValueError("The argument parameter should be list or "
                         "tuple with at least one element.")

    samples = arguments[0]
    n_samples = len(samples)
    batch_iterator = iter_batches(n_samples, batch_size)

    if show_progressbar:
        batch_iterator = logger.progressbar(list(batch_iterator),
                                            description=description,
                                            file=logger.stdout)

    outputs = []
    for batch in batch_iterator:
        sliced_arguments = [argument[batch] for argument in arguments]

        output = function(*sliced_arguments)
        outputs.append(output)

        if show_progressbar and logger.enable and show_error_output:
            batch_iterator.show_in_next_iteration = {
                'error': format_error(output)
            }

    return outputs


def average_batch_errors(errors, n_samples, batch_size):
    """
    Computes average error per sample.

    Parameters
    ----------
    errors : list
        List of errors where each element is a average error
        per batch.

    n_samples : int
        Number of samples in the dataset.

    batch_size : int
        Mini-batch size.

    Returns
    -------
    float
        Average error per sample.
    """
    if batch_size is None:
        return errors[0]

    n_samples_in_final_batch = n_samples % batch_size

    if n_samples_in_final_batch == 0:
        return batch_size * sum(errors) / n_samples

    all_errors_without_last = errors[:-1]
    last_error = errors[-1]

    total_error = (
        sum(all_errors_without_last) * batch_size +
        last_error * n_samples_in_final_batch
    )
    average_error = total_error / n_samples
    return average_error


class MinibatchTrainingMixin(Configurable):
    """
    Mixin that helps to train network using mini-batches.

    Notes
    -----
    Works with ``BaseNetwork`` class.

    Parameters
    ----------
    batch_size : int or {{None, -1, 'all', '*', 'full'}}
        Set up min-batch size. If mini-batch size is equal
        to one of the values from the list (like ``full``) then
        it's just a batch that equal to number of samples.
        Defaults to ``128``.
    """
    batch_size = BatchSizeProperty(default=128)

    def apply_batches(self, function, input_data, arguments=(), description='',
                      show_progressbar=None, show_error_output=False):
        """
        Apply function per each mini-batch.

        Parameters
        ----------
        function : callable

        input_data : array-like
            First argument to the function that can be divided
            into mini-batches.

        arguments : tuple
            Additional arguments to the function.

        description : str
            Some description for the progressbar. Defaults to ``''``.

        show_progressbar : None or bool
            ``True``/``False`` will show/hide progressbar. If value
            is equal to ``None`` than progressbar will be visible in
            case if network expects to see logging after each
            training epoch.

        show_error_output : bool
            Assumes that outputs from the function errors.
            ``True`` will show information in the progressbar.
            Error will be related to the last epoch.

        Returns
        -------
        list
            List of outputs from the function. Each output is an
            object that ``function`` returned.
        """
        arguments = as_tuple(input_data, arguments)

        if cannot_divide_into_batches(input_data, self.batch_size):
            return [function(*arguments)]

        if show_progressbar is None:
            show_progressbar = (self.training and
                                self.training.show_epoch == 1)

        return apply_batches(
            function=function,
            arguments=arguments,
            batch_size=self.batch_size,

            description=description,
            logger=self.logs,
            show_progressbar=show_progressbar,
            show_error_output=show_error_output,
        )


def count_samples(input_data):
    """
    Count number of samples in the input data

    Parameters
    ----------
    input_data : array-like or list/tuple of array-like objects
        Input data to the network

    Returns
    -------
    int
        Number of samples in the input data.
    """
    if isinstance(input_data, (list, tuple)):
        return len(input_data[0])
    return len(input_data)


class MinibatchGradientDescent(GradientDescent, MinibatchTrainingMixin):
    """
    Mini-batch Gradient Descent algorithm.

    Parameters
    ----------
    {MinibatchTrainingMixin.Parameters}

    {GradientDescent.Parameters}

    Attributes
    ----------
    {GradientDescent.Attributes}

    Methods
    -------
    {GradientDescent.Methods}

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

    def train_epoch(self, input_train, target_train):
        """
        Train one epoch.

        Parameters
        ----------
        input_train : array-like
            Training input dataset.

        target_train : array-like
            Training target dataset.

        Returns
        -------
        float
            Training error.
        """
        errors = self.apply_batches(
            function=self.methods.train_epoch,
            input_data=input_train,
            arguments=as_tuple(target_train),

            description='Training batches',
            show_error_output=True,
        )
        return average_batch_errors(
            errors,
            n_samples=count_samples(input_train),
            batch_size=self.batch_size,
        )

    def prediction_error(self, input_data, target_data):
        """
        Check the prediction error for the specified input samples
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

        errors = self.apply_batches(
            function=self.methods.prediction_error,
            input_data=input_data,
            arguments=as_tuple(target_data),

            description='Validation batches',
            show_error_output=True,
        )
        return average_batch_errors(
            errors,
            n_samples=count_samples(input_data),
            batch_size=self.batch_size,
        )

    def predict(self, input_data):
        """
        Makes a raw prediction.

        Parameters
        ----------
        input_data : array-like

        Returns
        -------
        array-like
        """
        outputs = self.apply_batches(
            function=self.methods.predict,
            input_data=self.format_input_data(input_data),

            description='Prediction batches',
            show_progressbar=True,
            show_error_output=False,
        )
        return np.concatenate(outputs, axis=0)
