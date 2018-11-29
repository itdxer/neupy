import abc
import time
from functools import wraps

import six
import tensorflow as tf

from neupy import layers
from neupy.layers.utils import preformat_layer_shape
from neupy.layers.connections import LayerConnection, is_sequential
from neupy.layers.connections.base import create_input_variables
from neupy.exceptions import InvalidConnection
from neupy.core.properties import FunctionWithOptionsProperty
from neupy.algorithms.base import BaseNetwork
from neupy.utils import (
    AttributeKeyDict, asfloat, format_data, as_tuple,
    tensorflow_session, initialize_uninitialized_variables
)
from .gd import errors


__all__ = ('ConstructibleNetwork',)


def does_layer_accept_1d_feature(layer):
    """
    Check if 1D feature values are valid for the layer.

    Parameters
    ----------
    layer : object

    Returns
    -------
    bool
    """
    return (layer.output_shape == (1,))


def generate_layers(layers_sizes):
    """
    Create from list of layer sizes basic linear network.

    Parameters
    ----------
    layers_sizes : list or tuple
        Ordered list of network connection structure.

    Returns
    -------
    LayerConnection
        Constructed connection.
    """
    layers_sizes = list(layers_sizes)
    n_layers = len(layers_sizes)

    if n_layers <= 1:
        raise ValueError("Cannot generate network that "
                         "has less than two layers")

    input_layer_size = layers_sizes.pop(0)
    connection = layers.Input(input_layer_size)

    for output_size in layers_sizes:
        next_layer = layers.Sigmoid(output_size)
        connection = LayerConnection(connection, next_layer)

    return connection


def clean_layers(connection):
    """
    Clean layers connections and format transform them into one format.
    Also this function validate layers connections.

    Parameters
    ----------
    connection : list, tuple or object
        Layers connetion in different formats.

    Returns
    -------
    object
        Cleaned layers connection.
    """
    if all(isinstance(element, int) for element in connection):
        connection = generate_layers(connection)

    if isinstance(connection, (list, tuple)):
        connection = layers.join(*connection)

    return connection


class BaseAlgorithm(six.with_metaclass(abc.ABCMeta)):
    """
    Base class for algorithms implemeted in Tensorflow.

    Attributes
    ----------
    variables : dict
        Tensorflow variables.

    methods : dict
        Compiled Tensorflow functions.
    """
    def __init__(self, *args, **kwargs):
        super(BaseAlgorithm, self).__init__(*args, **kwargs)

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

    @abc.abstractmethod
    def init_input_output_variables(self):
        """
        Initialize input and output Tensorflow variables.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def init_variables(self):
        """
        Initialize Tensorflow variables.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def init_methods(self):
        """
        Initialize Tensorflow functions.
        """
        raise NotImplementedError


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


class ConstructibleNetwork(BaseAlgorithm, BaseNetwork):
    """
    Class contains functionality that helps work with network that have
    constructible layers architecture.

    Parameters
    ----------
    connection : list, tuple or LayerConnection instance
        Network's architecture. There are a few ways
        to define it.

        - List of layers.
          For instance, ``[Input(2), Tanh(4), Relu(1)]``.

        - Construct layer connections.
          For instance, ``Input(2) > Tanh(4) > Relu(1)``.

        - Tuple of integers. Each integer defines Sigmoid layer
          and it's input size.  For instance,  value ``(2, 4, 1)``
          means that network has 3 layers with 2 input units,
          4 hidden units and 1 output unit.

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

    {BaseNetwork.Parameters}

    Attributes
    ----------
    {BaseNetwork.Attributes}

    Methods
    -------
    {BaseSkeleton.predict}

    train(input_train, target_train, input_test=None, target_test=None,\
    epochs=100, epsilon=None)
        Train network. You can control network's training procedure
        with ``epochs`` and ``epsilon`` parameters.
        The ``input_test`` and ``target_test`` should be presented
        both in case of you need to validate network's training
        after each iteration.

    {BaseSkeleton.fit}
    """
    error = FunctionWithOptionsProperty(default='mse', choices={
        'mae': errors.mae,
        'mse': errors.mse,
        'rmse': errors.rmse,
        'msle': errors.msle,
        'rmsle': errors.rmsle,

        'binary_crossentropy': errors.binary_crossentropy,
        'categorical_crossentropy': errors.categorical_crossentropy,

        'binary_hinge': errors.binary_hinge,
        'categorical_hinge': errors.categorical_hinge,
    })

    def __init__(self, connection, *args, **kwargs):
        self.connection = clean_layers(connection)

        self.layers = list(self.connection)
        graph = self.connection.graph

        if len(self.connection.output_layers) != 1:
            n_outputs = len(graph.output_layers)
            raise InvalidConnection("Connection should have one output "
                                    "layer, got {}".format(n_outputs))

        self.output_layer = graph.output_layers[0]

        super(ConstructibleNetwork, self).__init__(*args, **kwargs)

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

        self.variables.update(
            step=tf.Variable(
                asfloat(self.step),
                name='network/scalar-step',
                dtype=tf.float32
            ),
            epoch=tf.Variable(
                asfloat(self.last_epoch),
                name='network/scalar-epoch',
                dtype=tf.float32
            ),

            prediction_func=prediction,
            train_prediction_func=train_prediction,

            error_func=self.error(network_output, train_prediction),
            validation_error_func=self.error(network_output, prediction),
        )

    def init_methods(self):
        network_inputs = self.variables.network_inputs
        network_output = self.variables.network_output

        with tf.name_scope('training-updates'):
            training_updates = self.init_train_updates()

            for layer in self.layers:
                training_updates.extend(layer.updates)

        initialize_uninitialized_variables()

        self.methods.update(
            predict=function(
                inputs=network_inputs,
                outputs=self.variables.prediction_func,
                name='network/func-predict'
            ),
            train_epoch=function(
                inputs=network_inputs + [network_output],
                outputs=self.variables.error_func,
                updates=training_updates,
                name='network/func-train-epoch'
            ),
            prediction_error=function(
                inputs=network_inputs + [network_output],
                outputs=self.variables.validation_error_func,
                name='network/func-prediction-error'
            )
        )

    def init_train_updates(self):
        """
        Initialize updates that would be applied after
        each training epoch.
        """
        raise NotImplementedError()

    def format_input_data(self, input_data):
        """
        Input data format is depend on the input layer
        structure.

        Parameters
        ----------
        input_data : array-like or None

        Returns
        -------
        array-like or None
            Function returns formatted array.
        """
        input_layers = self.connection.input_layers

        if not isinstance(input_data, (tuple, list)):
            input_layer = input_layers[0]
            is_feature1d = does_layer_accept_1d_feature(input_layer)
            return format_data(input_data, is_feature1d)

        formated_data = []
        for input_to_layer, input_layer in zip(input_data, input_layers):
            is_feature1d = does_layer_accept_1d_feature(input_layer)
            data = format_data(input_to_layer, is_feature1d)
            formated_data.append(data)

        return tuple(formated_data)

    def format_target_data(self, target_data):
        """
        Target data format is depend on the output layer
        structure.

        Parameters
        ----------
        target_data : array-like or None

        Returns
        -------
        array-like or None
            Function returns formatted array.
        """
        return format_data(target_data)

    def prediction_error(self, input_data, target_data):
        """
        Calculate prediction accuracy for input data.

        Parameters
        ----------
        input_data : array-like
        target_data : array-like

        Returns
        -------
        float
            Prediction error.
        """
        return self.methods.prediction_error(*as_tuple(
            self.format_input_data(input_data),
            self.format_target_data(target_data)
        ))

    def predict(self, input_data):
        """
        Return prediction results for the input data.

        Parameters
        ----------
        input_data : array-like

        Returns
        -------
        array-like
        """
        input_data = self.format_input_data(input_data)
        return self.methods.predict(*as_tuple(input_data))

    def on_epoch_start_update(self, epoch):
        """
        Function will be triggered before the training epoch procedure.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        """
        super(ConstructibleNetwork, self).on_epoch_start_update(epoch)
        self.variables.epoch.load(epoch, tensorflow_session())

    def train(self, input_train, target_train, input_test=None,
              target_test=None, *args, **kwargs):
        """
        Train neural network.
        """
        is_test_data_partialy_missed = (
            (input_test is None and target_test is not None) or
            (input_test is not None and target_test is None)
        )

        if is_test_data_partialy_missed:
            raise ValueError("Input or target test samples are missed. They "
                             "must be defined together or none of them.")

        input_train = self.format_input_data(input_train)
        target_train = self.format_target_data(target_train)

        if input_test is not None:
            input_test = self.format_input_data(input_test)

        if target_test is not None:
            target_test = self.format_target_data(target_test)

        return super(ConstructibleNetwork, self).train(
            input_train=input_train, target_train=target_train,
            input_test=input_test, target_test=target_test,
            *args, **kwargs
        )

    def train_epoch(self, input_train, target_train):
        """
        Trains neural network over one epoch.

        Parameters
        ----------
        input_data : array-like
        target_data : array-like

        Returns
        -------
        float
            Prediction error.
        """
        return self.methods.train_epoch(*as_tuple(input_train, target_train))

    def architecture(self):
        """
        Shows network's architecture in the terminal if
        ``verbose`` parameter is equal to ``True``.
        """
        if not is_sequential(self.connection):
            raise TypeError("You can check architecture only for sequential "
                            "connections. For other types of connections "
                            "it's better to use the "
                            "`neupy.plots.network_structure` function.")

        self.logs.title("Network's architecture")

        values = []
        for index, layer in enumerate(self.layers, start=1):
            input_shape = preformat_layer_shape(layer.input_shape)
            output_shape = preformat_layer_shape(layer.output_shape)
            classname = layer.__class__.__name__

            values.append((index, input_shape, classname, output_shape))

        self.logs.table(values, headers=[
            '#', 'Input shape', 'Layer type', 'Output shape'
        ])
        self.logs.newline()

    def __repr__(self):
        return "{}({}, {})".format(self.class_name(), self.connection,
                                   self.repr_options())
