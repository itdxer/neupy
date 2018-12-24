from __future__ import division

import math
import time
from functools import wraps

import six
import numpy as np
import tensorflow as tf
import progressbar

from neupy import layers
from neupy.core.config import Configurable
from neupy.core.properties import (
    FunctionWithOptionsProperty,
    ScalarVariableProperty,
    BoundedProperty,
    Property,
)
from neupy.utils import (
    AttributeKeyDict, format_data, as_tuple,
    tensorflow_session, initialize_uninitialized_variables
)
from neupy.layers.utils import iter_parameters
from neupy.algorithms.gd import objectives
from neupy.layers.connections.base import create_input_variables
from neupy.exceptions import InvalidConnection
from neupy.algorithms.base import BaseNetwork


__all__ = ('BaseOptimizer', 'GradientDescent')


def function(inputs, outputs, updates=None, name=None):
    if updates is None:
        updates = []

    session = tensorflow_session()
    tensorflow_updates = []

    # Ensure that all new values has been computed. Absence of these
    # checks might lead to the non-deterministic update behaviour.
    new_values = [val[1] for val in updates if isinstance(val, (list, tuple))]

    # Make sure that all outputs has been computed
    with tf.control_dependencies(as_tuple(outputs, new_values)):
        for update in updates:
            if isinstance(update, (list, tuple)):
                old_value, new_value = update
                update = old_value.assign(new_value)
            tensorflow_updates.append(update)

        # Group variables in order to avoid output for the updates
        tensorflow_updates = tf.group(*tensorflow_updates)

    @wraps(function)
    def wrapper(*input_values):
        feed_dict = dict(zip(inputs, input_values))
        result, _ = session.run(
            [outputs, tensorflow_updates],
            feed_dict=feed_dict,
        )
        return result
    return wrapper


class BaseOptimizer(BaseNetwork):
    """
    Gradient descent algorithm.

    Parameters
    ----------
    connection : list, tuple or LayerConnection instance
        Network's architecture. There are a few ways
        to define it.

        - List of layers.
          For instance, ``[Input(2), Tanh(4), Relu(1)]``.

        - Constructed layers.
          For instance, ``Input(2) > Tanh(4) > Relu(1)``.

    regularizer : function or None
        Network's regularizer.

    error : str or function
        Error/loss function. Defaults to ``mse``.

        - ``mae`` - Mean Absolute Error.

        - ``mse`` - Mean Squared Error.

        - ``rmse`` - Root Mean Squared Error.

        - ``msle`` - Mean Squared Logarithmic Error.

        - ``rmsle`` - Root Mean Squared Logarithmic Error.

        - ``categorical_crossentropy`` - Categorical cross entropy.

        - ``binary_crossentropy`` - Binary cross entropy.

        - ``binary_hinge`` - Binary hinge entropy.

        - ``categorical_hinge`` - Categorical hinge entropy.

        - Custom function which accepts two mandatory arguments.
          The first one is expected value and the second one is
          predicted value. Example:

        .. code-block:: python

            def custom_func(expected, predicted):
                return expected - predicted

    step : float, Variable
        Learning rate, defaults to ``0.1``.

    {BaseNetwork.show_epoch}

    {BaseNetwork.shuffle_data}

    {BaseNetwork.epoch_end_signal}

    {BaseNetwork.verbose}

    Attributes
    ----------
    {BaseNetwork.Attributes}

    Methods
    -------
    {BaseSkeleton.predict}

    train(X_train, y_train, X_test=None, y_test=None, epochs=100)
        Train network. You can control network's training procedure
        with ``epochs`` parameter. The ``X_test`` and ``y_test`` should
        be presented both in case network's validation required
        after each training epoch.

    {BaseSkeleton.fit}
    """
    step = ScalarVariableProperty(default=0.1)
    regularizer = Property(default=None, allow_none=True)
    error = FunctionWithOptionsProperty(default='mse', choices={
        'mae': objectives.mae,
        'mse': objectives.mse,
        'rmse': objectives.rmse,
        'msle': objectives.msle,
        'rmsle': objectives.rmsle,

        'binary_crossentropy': objectives.binary_crossentropy,
        'categorical_crossentropy': objectives.categorical_crossentropy,

        'binary_hinge': objectives.binary_hinge,
        'categorical_hinge': objectives.categorical_hinge,
    })

    def __init__(self, connection, options=None, **kwargs):
        options = options or kwargs

        if isinstance(connection, (list, tuple)):
            connection = layers.join(*connection)

        self.connection = connection
        self.layers = list(self.connection)
        graph = self.connection.graph

        if len(self.connection.output_layers) != 1:
            n_outputs = len(graph.output_layers)
            raise InvalidConnection("Connection should have one output "
                                    "layer, got {}".format(n_outputs))

        super(BaseOptimizer, self).__init__(**options)

        self.logs.message(
            "TENSORFLOW",
            "Initializing Tensorflow variables and functions."
        )
        start_init_time = time.time()

        self.variables = AttributeKeyDict()
        self.methods = AttributeKeyDict()

        self.init_input_output_variables()
        self.init_variables()
        self.init_methods()

        finish_init_time = time.time()
        self.logs.message(
            "TENSORFLOW",
            "Initialization finished successfully. It took {:.2f} seconds"
            "".format(finish_init_time - start_init_time))

    def iter_params_and_grads(self):
        layers, parameters = [], []

        for layer, _, parameter in iter_parameters(self.layers):
            layers.append(layer)
            parameters.append(parameter)

        gradients = tf.gradients(self.variables.error_func, parameters)
        iterator = zip(layers, parameters, gradients)

        for layer, parameter, gradient in iterator:
            yield layer, parameter, gradient

    def init_train_updates(self):
        updates = []
        step = self.variables.step

        for layer, parameter, gradient in self.iter_params_and_grads():
            updates.append((parameter, parameter - step * gradient))

        return updates

    def init_input_output_variables(self):
        output_layer = self.connection.output_layers[0]
        self.variables.update(
            network_inputs=create_input_variables(
                self.connection.input_layers
            ),
            network_output=tf.placeholder(
                tf.float32,
                name='network-output/from-layer-{}'.format(output_layer.name),
            ),
        )

    def init_variables(self):
        network_inputs = self.variables.network_inputs
        network_output = self.variables.network_output

        train_prediction = self.connection.output(*network_inputs)
        with self.connection.disable_training_state():
            prediction = self.connection.output(*network_inputs)

        loss = self.error(network_output, train_prediction)
        val_loss = self.error(network_output, prediction)

        if self.regularizer is not None:
            loss = loss + self.regularizer(self.connection)

        self.variables.update(
            step=self.step,
            prediction_func=prediction,
            train_prediction_func=train_prediction,

            error_func=loss,
            validation_error_func=val_loss,
        )

    def init_methods(self):
        network_inputs = self.variables.network_inputs
        network_output = self.variables.network_output

        with tf.name_scope('training-updates'):
            training_updates = self.init_train_updates()

            for layer in self.layers:
                training_updates.extend(layer.updates)

            for variable in self.variables.values():
                if hasattr(variable, 'updates'):
                    training_updates.extend(variable.updates)

        initialize_uninitialized_variables()

        self.methods.update(
            predict=function(
                inputs=network_inputs,
                outputs=self.variables.prediction_func,
                name='network/func-predict'
            ),
            one_training_update=function(
                inputs=network_inputs + [network_output],
                outputs=self.variables.error_func,
                updates=training_updates,
                name='network/func-train-epoch'
            ),
            score=function(
                inputs=network_inputs + [network_output],
                outputs=self.variables.validation_error_func,
                name='network/func-prediction-error'
            )
        )

    def score(self, X, y):
        """
        Calculate prediction accuracy for input data.

        Parameters
        ----------
        X : array-like
        y : array-like

        Returns
        -------
        float
            Prediction error.
        """
        X_and_y = [format_data(x) for x in as_tuple(X, y)]
        return self.methods.score(*X_and_y)

    def predict(self, X):
        """
        Return prediction results for the input data.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        array-like
        """
        X = [format_data(x) for x in as_tuple(X)]
        return self.methods.predict(*X)

    def train(self, X_train, y_train, X_test=None, y_test=None,
              *args, **kwargs):

        is_test_data_partialy_missing = (
            (X_test is None and y_test is not None) or
            (X_test is not None and y_test is None)
        )

        if is_test_data_partialy_missing:
            raise ValueError("Input or target test samples are missed. They "
                             "must be defined together or none of them.")

        X_train = [format_data(x) for x in as_tuple(X_train)]
        y_train = format_data(y_train)

        if X_test is not None:
            X_test = [format_data(x) for x in as_tuple(X_test)]
            y_test = format_data(y_test)

        return super(BaseOptimizer, self).train(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            *args, **kwargs
        )

    def one_training_update(self, X_train, y_train):
        return self.methods.one_training_update(*as_tuple(X_train, y_train))

    def get_params(self, deep=False, with_connection=True):
        params = super(BaseOptimizer, self).get_params()
        if with_connection:
            params['connection'] = self.connection
        return params

    def __reduce__(self):
        parameters = self.get_params(with_connection=False)
        args = (self.connection, parameters)
        return (self.__class__.__name__, args)

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__,
            self.connection,
            self.repr_options())


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


def apply_batches(function, arguments, batch_size, description='',
                  show_progressbar=False, show_error_output=True,
                  scalar_output=True):
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

    if not scalar_output and show_error_output:
        raise ValueError("Cannot show error when output isn't scalar")

    samples = arguments[0]
    n_samples = len(samples)
    batch_iterator = list(iter_batches(n_samples, batch_size))
    bar = progressbar.NullBar()

    if show_progressbar:
        widgets = [
            progressbar.Timer(format='Time: %(elapsed)s'),
            ' |',
            progressbar.Percentage(),
            progressbar.Bar(),
            ' ',
            progressbar.ETA(),
        ]

        if show_error_output:
            widgets.extend([' | ', progressbar.DynamicMessage('error')])

        bar = progressbar.ProgressBar(
            widgets=widgets,
            max_value=len(batch_iterator),
            poll_interval=0.1,
        )
        bar.update(0)

    outputs = []
    for i, batch in enumerate(batch_iterator):
        sliced_arguments = [argument[batch] for argument in arguments]
        output = function(*sliced_arguments)

        if scalar_output:
            output = np.atleast_1d(output)

            if output.size > 1:
                raise ValueError(
                    "Cannot convert output from the batch, "
                    "because it has more than one output value")

            output = output.item(0)

        outputs.append(output)
        kwargs = {'error': output} if show_error_output else {}
        bar.update(i, **kwargs)

    bar.fd.write('\r' + ' ' * bar.term_width + '\r')
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

    def apply_batches(self, function, X, arguments=(), description='',
                      show_progressbar=None, show_error_output=False,
                      scalar_output=True):
        """
        Apply function per each mini-batch.

        Parameters
        ----------
        function : callable

        X : array-like
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

        scalar_output : bool
            ``True`` means that we expect scalar value per each

        Returns
        -------
        list
            List of outputs from the function. Each output is an
            object that ``function`` returned.
        """
        arguments = as_tuple(X, arguments)

        if cannot_divide_into_batches(X, self.batch_size):
            output = function(*arguments)
            if scalar_output:
                output = np.atleast_1d(output).item(0)
            return [output]

        if show_progressbar is None:
            show_progressbar = self.show_epoch == 1 and self.logs.enable

        return apply_batches(
            function=function,
            arguments=arguments,
            batch_size=self.batch_size,

            description=description,
            show_progressbar=show_progressbar,
            show_error_output=show_error_output,
            scalar_output=scalar_output,
        )


def count_samples(X):
    """
    Count number of samples in the input data

    Parameters
    ----------
    X : array-like or list/tuple of array-like objects
        Input data to the network

    Returns
    -------
    int
        Number of samples in the input data.
    """
    if isinstance(X, (list, tuple)):
        return len(X[0])
    return len(X)


class GradientDescent(BaseOptimizer, MinibatchTrainingMixin):
    """
    Mini-batch Gradient Descent algorithm.

    Parameters
    ----------
    {MinibatchTrainingMixin.Parameters}

    {BaseOptimizer.Parameters}

    Attributes
    ----------
    {BaseOptimizer.Attributes}

    Methods
    -------
    {BaseOptimizer.Methods}

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms
    >>> from neupy.algorithms import *
    >>>
    >>> x_train = np.array([[1, 2], [3, 4]])
    >>> y_train = np.array([[1], [0]])
    >>>
    >>> network = Input(2) > Sigmoid(3) > Sigmoid(1)
    >>> mgdnet = algorithms.GradientDescent(network, batch_size=1)
    >>> mgdnet.train(x_train, y_train)
    """
    def one_training_update(self, X_train, y_train):
        """
        Train one epoch.

        Parameters
        ----------
        X_train : array-like
            Training input dataset.

        y_train : array-like
            Training target dataset.

        Returns
        -------
        float
            Training error.
        """
        errors = self.apply_batches(
            function=self.methods.one_training_update,
            X=X_train,
            arguments=as_tuple(y_train),

            description='Training batches',
            show_error_output=True,
        )
        return average_batch_errors(
            errors,
            n_samples=count_samples(X_train),
            batch_size=self.batch_size,
        )

    def score(self, X, y):
        """
        Check the prediction error for the specified input samples
        and their targets.

        Parameters
        ----------
        X : array-like
        y : array-like

        Returns
        -------
        float
            Prediction error.
        """
        X = [format_data(x) for x in as_tuple(X)]
        y = format_data(y)

        errors = self.apply_batches(
            function=self.methods.score,
            X=X,
            arguments=as_tuple(y),

            description='Validation batches',
            show_error_output=True,
        )
        return average_batch_errors(
            errors,
            n_samples=count_samples(X),
            batch_size=self.batch_size,
        )

    def predict(self, X):
        """
        Makes a raw prediction.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        array-like
        """
        outputs = self.apply_batches(
            function=self.methods.predict,
            X=X,

            description='Prediction batches',
            show_progressbar=True,
            show_error_output=False,
            scalar_output=False,
        )
        return np.concatenate(outputs, axis=0)
