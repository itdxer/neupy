import abc
import time
import types

import six
import theano
import theano.sparse
import theano.tensor as T

from neupy import layers
from neupy.utils import AttributeKeyDict, asfloat, format_data
from neupy.layers.utils import preformat_layer_shape
from neupy.layers.connections import LayerConnection, is_sequential
from neupy.exceptions import InvalidConnection
from neupy.helpers import table
from neupy.core.properties import ChoiceProperty
from neupy.algorithms.base import BaseNetwork
from neupy.algorithms.utils import parameter_values
from neupy.algorithms.learning import SupervisedLearningMixin
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

    if isinstance(connection, tuple):
        connection = list(connection)

    if all(isinstance(element, int) for element in connection):
        connection = generate_layers(connection)

    islist = isinstance(connection, list)
    layer_types = (layers.BaseLayer, LayerConnection)

    if islist and isinstance(connection[0], layer_types):
        connection = layers.join(*connection)

    return connection


def create_input_variable(input_layer, name):
    """
    Create input variable based on input layer information.

    Parameters
    ----------
    input_layer : object
    name : str

    Returns
    -------
    Theano variable
    """
    dim_to_variable_type = {
        2: T.matrix,
        3: T.tensor3,
        4: T.tensor4,
    }

    # Shape doesn't include batch size dimension,
    # that's why we need to add one
    ndim = len(input_layer.input_shape) + 1

    if ndim not in dim_to_variable_type:
        raise ValueError("Layer's input needs to be 2, 3 or 4 dimensional. "
                         "Found {} dimensions".format(ndim))

    variable_type = dim_to_variable_type[ndim]
    return variable_type(name)


def create_output_variable(error_function, name):
    """
    Create output variable based on error function.

    Parameters
    ----------
    error_function : function
    name : str

    Returns
    -------
    Theano variable
    """
    # TODO: Solution is not user friendly. I need to find
    # better solution later.
    if hasattr(error_function, 'expected_dtype'):
        network_output_dtype = error_function.expected_dtype
    else:
        network_output_dtype = T.matrix

    return network_output_dtype(name)


class ErrorFunctionProperty(ChoiceProperty):
    """
    Property that helps select error function from
    available or define a new one.

    Parameters
    ----------
    {ChoiceProperty.Parameters}
    """
    def __set__(self, instance, value):
        if isinstance(value, types.FunctionType):
            return super(ChoiceProperty, self).__set__(instance, value)
        return super(ErrorFunctionProperty, self).__set__(instance, value)

    def __get__(self, instance, value):
        founded_value = super(ChoiceProperty, self).__get__(instance, value)
        if isinstance(founded_value, types.FunctionType):
            return founded_value
        return super(ErrorFunctionProperty, self).__get__(instance,
                                                          founded_value)


class BaseAlgorithm(six.with_metaclass(abc.ABCMeta)):
    """
    Base class for algorithms implemeted in Theano.

    Attributes
    ----------
    variables : dict
        Theano variables.

    methods : dict
        Compiled Theano functions.
    """
    def __init__(self, *args, **kwargs):
        super(BaseAlgorithm, self).__init__(*args, **kwargs)

        self.logs.message("THEANO", "Initializing Theano variables and "
                                    "functions.")
        start_init_time = time.time()

        self.variables = AttributeKeyDict()
        self.methods = AttributeKeyDict()

        self.init_input_output_variables()
        self.init_variables()
        self.init_methods()

        finish_init_time = time.time()
        self.logs.message("THEANO", "Initialization finished successfully. "
                          "It took {:.2f} seconds"
                          "".format(finish_init_time - start_init_time))

    @abc.abstractmethod
    def init_input_output_variables(self):
        """
        Initialize input and output Theano variables.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def init_variables(self):
        """
        Initialize Theano variables.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def init_methods(self):
        """
        Initialize Theano functions.
        """
        raise NotImplementedError


class ConstructibleNetwork(SupervisedLearningMixin, BaseAlgorithm,
                           BaseNetwork):
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

    {SupervisedLearningMixin.train}

    {BaseSkeleton.fit}
    """
    error = ErrorFunctionProperty(default='mse', choices={
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

        if len(self.connection.input_layers) != 1:
            n_inputs = len(graph.input_layers)
            raise InvalidConnection("Connection should have one input "
                                    "layer, got {}".format(n_inputs))

        if len(self.connection.output_layers) != 1:
            n_outputs = len(graph.output_layers)
            raise InvalidConnection("Connection should have one output "
                                    "layer, got {}".format(n_outputs))

        self.input_layer = graph.input_layers[0]
        self.output_layer = graph.output_layers[0]

        super(ConstructibleNetwork, self).__init__(*args, **kwargs)

    def init_input_output_variables(self):
        self.variables.update(
            network_input=create_input_variable(
                self.input_layer,
                name='algo:network/var:network-input',
            ),
            network_output=create_output_variable(
                self.error,
                name='algo:network/var:network-output',
            ),
        )

    def init_variables(self):
        network_input = self.variables.network_input
        network_output = self.variables.network_output

        train_prediction = self.connection.output(network_input)
        with self.connection.disable_training_state():
            prediction = self.connection.output(network_input)

        self.variables.update(
            step=theano.shared(
                name='algo:network/scalar:step',
                value=asfloat(self.step)
            ),
            epoch=theano.shared(
                name='algo:network/scalar:epoch',
                value=asfloat(self.last_epoch)
            ),

            prediction_func=prediction,
            train_prediction_func=train_prediction,

            error_func=self.error(network_output, train_prediction),
            validation_error_func=self.error(network_output, prediction),
        )

    def init_methods(self):
        network_input = self.variables.network_input
        network_output = self.variables.network_output

        self.methods.update(
            predict=theano.function(
                inputs=[network_input],
                outputs=self.variables.prediction_func,
                name='algo:network/func:predict'
            ),
            train_epoch=theano.function(
                inputs=[network_input, network_output],
                outputs=self.variables.error_func,
                updates=self.init_train_updates(),
                name='algo:network/func:train-epoch'
            ),
            prediction_error=theano.function(
                inputs=[network_input, network_output],
                outputs=self.variables.validation_error_func,
                name='algo:network/func:prediction-error'
            )
        )

    def init_train_updates(self):
        """
        Initialize train function update in Theano format that
        would be trigger after each training epoch.
        """
        updates = []
        for layer in self.layers:
            updates.extend(self.init_layer_updates(layer))
        return updates

    def init_layer_updates(self, layer):
        """
        Initialize train function update in Theano format that
        would be trigger after each training epoch for each layer.

        Parameters
        ----------
        layer : object
            Any layer that inherit from layers.BaseLayer class.

        Returns
        -------
        list
            Update that excaptable by ``theano.function``. There should be
            a lits that contains tuples with 2 elements. First one should
            be parameter that would be updated after epoch and the second one
            should update rules for this parameter. For example parameter
            could be a layer's weight and bias.
        """
        updates = []
        for parameter in parameter_values(layer):
            updates.extend(self.init_param_updates(layer, parameter))
        updates.extend(layer.updates)
        return updates

    def init_param_updates(self, layer, parameter):
        """
        Initialize parameter updates.

        Parameters
        ----------
        layer : object
            Any layer that inherit from BaseLayer class.
        parameter : object
            Usualy it is a weight or bias.

        Returns
        -------
        list
            List of updates related to the specified parameter.
        """
        return []

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
        if input_data is not None:
            is_feature1d = does_layer_accept_1d_feature(self.input_layer)
            return format_data(input_data, is_feature1d)

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
        if target_data is not None:
            is_feature1d = does_layer_accept_1d_feature(self.output_layer)
            return format_data(target_data, is_feature1d)

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
        return self.methods.prediction_error(
            self.format_input_data(input_data),
            self.format_target_data(target_data)
        )

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
        return self.methods.predict(input_data)

    def on_epoch_start_update(self, epoch):
        """
        Function would be trigger before run all training procedure
        related to the current epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        """
        super(ConstructibleNetwork, self).on_epoch_start_update(epoch)
        self.variables.epoch.set_value(epoch)

    def train(self, input_train, target_train, input_test=None,
              target_test=None, *args, **kwargs):
        """
        Trains neural network.
        """
        return super(ConstructibleNetwork, self).train(
            self.format_input_data(input_train),
            self.format_target_data(target_train),
            self.format_input_data(input_test),
            self.format_target_data(target_test),
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
        return self.methods.train_epoch(input_train, target_train)

    def architecture(self):
        """
        Shows network's architecture in the terminal if
        ``verbose`` parameter is equal to ``True``.
        """
        if not is_sequential(self.connection):
            raise TypeError("You can check architecture only for sequential "
                            "connections. For other types of connections it's "
                            "better to use the `neupy.plots.layer_structure` "
                            "function.")

        self.logs.title("Network's architecture")

        values = []
        for index, layer in enumerate(self.layers, start=1):
            input_shape = preformat_layer_shape(layer.input_shape)
            output_shape = preformat_layer_shape(layer.output_shape)
            classname = layer.__class__.__name__

            values.append((index, input_shape, classname, output_shape))

        table.TableBuilder.show_full_table(
            columns=[
                table.Column(name="#"),
                table.Column(name="Input shape"),
                table.Column(name="Layer Type"),
                table.Column(name="Output shape"),
            ],
            values=values,
            stdout=self.logs.write,
        )
        self.logs.newline()

    def __repr__(self):
        return "{}({}, {})".format(self.class_name(), self.connection,
                                   self.repr_options())
