import time
import types

import theano
import theano.sparse
import theano.tensor as T

from neupy.utils import (AttributeKeyDict, asfloat, is_list_of_integers,
                         format_data, does_layer_accept_1d_feature)
from neupy.layers import BaseLayer, Output, Dropout
from neupy.layers.utils import generate_layers
from neupy.core.properties import ChoiceProperty
from neupy.layers.connections import LayerConnection, NetworkConnectionError
from neupy.network import errors
from .learning import SupervisedLearning
from .base import BaseNetwork


__all__ = ('ConstructableNetwork',)


def clean_layers(connection):
    """ Clean layers connections and format transform them into one format.
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

    if is_list_of_integers(connection):
        connection = generate_layers(list(connection))

    if isinstance(connection, tuple):
        connection = list(connection)

    islist = isinstance(connection, list)

    if islist and isinstance(connection[0], BaseLayer):
        chain_connection = connection.pop()
        for layer in reversed(connection):
            chain_connection = LayerConnection(layer, chain_connection)
        connection = chain_connection

    elif islist and isinstance(connection[0], LayerConnection):
        pass

    if not isinstance(connection.output_layer, Output):
        raise NetworkConnectionError("Final layer must be Output class "
                                     "instance.")

    return connection


def create_input_variable(input_layer, variable_name):
    """ Create input variable based on input layer information.

    Parameters
    ----------
    input_layer : object
    variable_name : str

    Returns
    -------
    Theano variable
    """
    dim_to_variable_type = {
        2: T.matrix,
        3: T.tensor3,
        4: T.tensor4,
    }
    ndim = input_layer.weight.ndim

    if ndim not in dim_to_variable_type:
        raise ValueError("Layer's input needs to be 2, 3 or 4 dimensional. "
                         "Found {}".format(ndim))

    variable_type = dim_to_variable_type[ndim]
    return variable_type(variable_name)


def create_output_variable(error_function, variable_name):
    """ Create output variable based on error function.

    Parameters
    ----------
    error_function : function
    variable_name : str

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

    return network_output_dtype(variable_name)


def find_input_layer(layers):
    """ Function checks list of layer and finds an input layer.

    Parameters
    ----------
    layers : iterative object
        Ordered list of layers.

    Returns
    -------
    BaseLayer instance or None
        Function will return input layer if it exists in the specified
        connection structure. Otherwise, output will be equal to ``None``.
    """
    for layer in layers:
        if not isinstance(layer, Dropout):
            return layer


class ErrorFunctionProperty(ChoiceProperty):
    """ Property that helps select error function from
    available or define a new one.

    Parameters
    ----------
    {ChoiceProperty.choices}
    {BaseProperty.default}
    {BaseProperty.required}
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


class ConstructableNetwork(SupervisedLearning, BaseNetwork):
    """ Class contains functionality that helps work with network that have
    constructable layers architecture.

    Parameters
    ----------
    connection : list, tuple or object
        Network architecture. That variables could be described in
        different ways. The simples one is a list or tuple that contains
        integers. Each integer describe layer input size. For example,
        ``(2, 4, 1)`` means that network will have 3 layers with 2 input
        units, 4 hidden units and 1 output unit. The one limitation of that
        method is that all layers automaticaly would with sigmoid actiavtion
        function. Other way is just a list of ``BaseLayer``` class
        instances. For example: ``[Tanh(2), Relu(4), Output(1)].
        And the most readable one is just layer pipeline
        ``Tanh(2) > Relu(4) > Output(1)``.
    error : {{'mse', 'rmse', 'mae', 'categorical_crossentropy', \
    'binary_crossentropy'}} or function
        Function that calculate prediction error.
        Defaults to ``mse``.

        * ``mae`` - Mean Absolute Error.

        * ``mse`` - Mean Squared Error.

        * ``rmse`` - Root Mean Squared Error.

        * ``msle`` - Mean Squared Logarithmic Error.

        * ``rmsle`` - Root Mean Squared Logarithmic Error.

        * ``categorical_crossentropy`` - Categorical cross entropy.

        * ``binary_crossentropy`` - Binary cross entropy.

        * Custom function that accept two mandatory arguments.
        The first one is expected value and the second one is
        predicted value. Example: ``custom_func(expected, predicted)``
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
    """
    error = ErrorFunctionProperty(default='mse', choices={
        'mae': errors.mae,
        'mse': errors.mse,
        'rmse': errors.rmse,
        'msle': errors.msle,
        'rmsle': errors.rmsle,
        'binary_crossentropy': errors.binary_crossentropy,
        'categorical_crossentropy': errors.categorical_crossentropy,
    })

    def __init__(self, connection, *args, **kwargs):
        self.connection = clean_layers(connection)

        self.all_layers = list(self.connection)
        self.layers = self.all_layers[:-1]

        self.input_layer = find_input_layer(self.layers)
        self.hidden_layers = self.layers[1:]
        self.output_layer = self.all_layers[-1]

        self.init_layers()
        super(ConstructableNetwork, self).__init__(*args, **kwargs)

        self.logs.message("THEANO", "Initializing Theano variables and "
                                    "functions.")
        start_init_time = time.time()

        self.variables = AttributeKeyDict(
            network_input=create_input_variable(
                self.input_layer, variable_name='x'
            ),
            network_output=create_output_variable(
                self.error, variable_name='y'
            ),
        )
        self.methods = AttributeKeyDict()

        self.init_variables()
        self.init_methods()

        finish_init_time = time.time()
        self.logs.message("THEANO", "Initialization finished sucessfully. "
                          "It took {:.2f} seconds"
                          "".format(finish_init_time - start_init_time))

    def init_variables(self):
        """ Initialize Theano variables.
        """
        network_input = self.variables.network_input
        network_output = self.variables.network_output

        train_prediction = prediction = network_input
        for layer in self.layers:
            if not isinstance(layer, Dropout):
                prediction = layer.output(prediction)
            train_prediction = layer.output(train_prediction)

        self.variables.update(
            step=theano.shared(name='step', value=asfloat(self.step)),
            epoch=theano.shared(name='epoch', value=asfloat(self.last_epoch)),

            prediction_func=prediction,
            train_prediction_func=train_prediction,

            error_func=self.error(network_output, train_prediction),
            validation_error_func=self.error(network_output, prediction),
        )

    def init_methods(self):
        """ Initialize all methods that needed for prediction and
        training procedures.
        """
        network_input = self.variables.network_input
        network_output = self.variables.network_output

        self.methods.predict_raw = theano.function(
            inputs=[self.variables.network_input],
            outputs=self.variables.prediction_func
        )
        self.methods.train_epoch = theano.function(
            inputs=[network_input, network_output],
            outputs=self.variables.error_func,
            updates=self.init_train_updates(),
        )
        self.methods.prediction_error = theano.function(
            inputs=[network_input, network_output],
            outputs=self.variables.validation_error_func
        )

    def init_layers(self):
        """ Initialize layers in the same order as they were list in
        network initialization step.
        """
        for layer in self.layers:
            layer.initialize()

    def init_train_updates(self):
        """ Initialize train function update in Theano format that
        would be trigger after each training epoch.
        """
        updates = []
        for layer in self.layers:
            updates.extend(self.init_layer_updates(layer))
        return updates

    def init_layer_updates(self, layer):
        """ Initialize train function update in Theano format that
        would be trigger after each training epoch for each layer.

        Parameters
        ----------
        layer : object
            Any layer that inherit from BaseLayer class.

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
        for parameter in layer.parameters:
            updates.extend(self.init_param_updates(layer, parameter))
        return updates

    def init_param_updates(self, parameter):
        """ Initialize parameter updates.

        Parameters
        ----------
        parameter : object
            Usualy it is a weight or bias.

        Returns
        -------
        list
            List of updates related to the specified parameter.
        """
        return []

    def format_input_data(self, input_data):
        """ Input data format is depence on the input layer
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
        """ Target data format is depence on the output layer
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
        """ Calculate prediction accuracy for input data.

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

    def predict_raw(self, input_data):
        """ Make raw prediction without final layer postprocessing step.

        Parameters
        ----------
        input_data : array-like

        Returns
        -------
        array-like
        """
        input_data = self.format_input_data(input_data)
        return self.methods.predict_raw(input_data)

    def predict(self, input_data):
        """ Return prediction results for the input data. Output result also
        include postprocessing step related to the final layer that
        transform output to convenient format for end-use.

        Parameters
        ----------
        input_data : array-like

        Returns
        -------
        array-like
        """
        raw_prediction = self.predict_raw(input_data)
        return self.output_layer.output(raw_prediction)

    def on_epoch_start_update(self, epoch):
        """ Function would be trigger before run all training procedure
        related to the current epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        """
        super(ConstructableNetwork, self).on_epoch_start_update(epoch)
        self.variables.epoch.set_value(epoch)

    def train(self, input_train, target_train, input_test=None,
              target_test=None, *args, **kwargs):
        """ Trains neural network.
        """
        return super(ConstructableNetwork, self).train(
            self.format_input_data(input_train),
            self.format_target_data(target_train),
            self.format_input_data(input_test),
            self.format_target_data(target_test),
            *args, **kwargs
        )

    def train_epoch(self, input_train, target_train):
        """ Trains neural network over one epoch.

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
        """ Shows network's architecture in the terminal if
        ``verbose`` parameter is equal to ``True``.
        """
        self.logs.title("Network's architecture")

        for i, layer in enumerate(self.all_layers, start=1):
            self.logs.write("{:>3}. {}".format(i, str(layer)))

        self.logs.newline()

    def __repr__(self):
        return "{}({}, {})".format(self.class_name(), self.connection,
                                   self._repr_options())
