import theano
import theano.tensor as T


from neupy.utils import (AttributeKeyDict, asfloat, is_list_of_integers,
                         format_data, is_layer_accept_1d_feature)
from neupy.layers import BaseLayer, Output, Dropout
from neupy.layers.utils import generate_layers
from neupy.core.properties import ChoiceProperty
from neupy.layers.connections import LayerConnection, NetworkConnectionError
from neupy.network import errors
from .learning import SupervisedLearning
from .base import BaseNetwork


__all__ = ('ConstructableNetwork', 'SupervisedConstructableNetwork')


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


class ConstructableNetwork(BaseNetwork):
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
    {BaseNetwork.step}
    {BaseNetwork.show_epoch}
    {BaseNetwork.shuffle_data}
    {BaseNetwork.epoch_end_signal}
    {BaseNetwork.train_end_signal}
    {Verbose.verbose}

    Methods
    -------
    {BaseNetwork.plot_errors}
    {BaseNetwork.last_error}
    {BaseNetwork.last_validation_error}
    {BaseNetwork.previous_error}
    """

    def __init__(self, connection, *args, **kwargs):
        self.connection = clean_layers(connection)

        self.layers = list(self.connection)
        self.input_layer = self.layers[0]
        self.hidden_layers = self.layers[1:-1]
        self.output_layer = self.layers[-1]
        self.train_layers = self.layers[:-1]

        self.init_layers()
        super(ConstructableNetwork, self).__init__(*args, **kwargs)

        self.variables = AttributeKeyDict(
            network_input=T.matrix('x'),
            network_output=T.matrix('y'),
        )
        self.methods = AttributeKeyDict()

        self.init_variables()
        self.init_methods()

    def init_variables(self):
        """ Initialize Theano variables.
        """

        network_input = self.variables.network_input

        train_layer_input = layer_input = network_input
        for layer in self.train_layers:
            if not isinstance(layer, Dropout):
                layer_input = layer.output(layer_input)
            train_layer_input = layer.output(train_layer_input)

        self.variables.update(
            step=theano.shared(name='step', value=asfloat(self.step)),
            epoch=theano.shared(name='epoch', value=1, borrow=False),
            prediction_func=layer_input,
            train_prediction_func=train_layer_input,
        )

    def init_methods(self):
        """ Initialize all methods that needed for prediction and
        training procedures.
        """
        self.methods.predict_raw = theano.function(
            inputs=[self.variables.network_input],
            outputs=self.variables.prediction_func
        )

    def init_layers(self):
        """ Initialize layers in the same order as they were list in
        network initialization step.
        """
        for layer in self.train_layers:
            layer.initialize()

    def init_train_updates(self):
        """ Initialize train function update in Theano format that
        would be trigger after each trainig epoch.
        """
        updates = []
        for layer in self.train_layers:
            updates.extend(self.init_layer_updates(layer))
        return updates

    def init_layer_updates(self, layer):
        """ Initialize train function update in Theano format that
        would be trigger after each trainig epoch for each layer.

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
        return []

    def prediction_error(self, input_data, target_data):
        """ Calculate prediction accuracy for input data.
        """
        input_data = format_data(input_data)
        return self.methods.prediction_error(input_data, target_data)

    def predict_raw(self, input_data):
        """ Make raw prediction without final layer postprocessing step.
        """
        is_feature1d = is_layer_accept_1d_feature(self.input_layer)
        input_data = format_data(input_data, is_feature1d)
        return self.methods.predict_raw(input_data)

    def predict(self, input_data):
        """ Return prediction results for the input data. Output result also
        include postprocessing step related to the final layer that
        transform output to convenient format for end-use.
        """
        raw_prediction = self.predict_raw(input_data)
        return self.output_layer.output(raw_prediction)

    def epoch_start_update(self, epoch):
        """ Function would be trigger before run all training procedure
        related to the current epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        """
        super(ConstructableNetwork, self).epoch_start_update(epoch)
        self.variables.epoch.set_value(epoch)

    def train_epoch(self, input_train, target_train):
        return self.methods.train_epoch(input_train, target_train)

    def __repr__(self):
        return "{}({}, {})".format(self.class_name(), self.connection,
                                   self._repr_options())


class SupervisedConstructableNetwork(SupervisedLearning, ConstructableNetwork):
    """ Constructuble Neural Network that contains supervised
    learning features.

    Parameters
    ----------
    error : str
        Function which controls your training error.
        Defaults to ``mse``
    {ConstructableNetwork.connection}
    {BaseNetwork.step}
    {BaseNetwork.show_epoch}
    {BaseNetwork.shuffle_data}
    {BaseNetwork.epoch_end_signal}
    {BaseNetwork.train_end_signal}
    {Verbose.verbose}

    Methods
    -------
    {BaseNetwork.plot_errors}
    {BaseNetwork.last_error}
    {BaseNetwork.last_validation_error}
    {BaseNetwork.previous_error}
    """

    error = ChoiceProperty(default='mse', choices={
        'mse': errors.mse,
        'binary_crossentropy': errors.binary_crossentropy,
        'categorical_crossentropy': errors.categorical_crossentropy,
    })

    def init_variables(self):
        """ Initialize Theano variables.
        """
        super(SupervisedConstructableNetwork, self).init_variables()

        network_output = self.variables.network_output
        prediction = self.variables.train_prediction_func

        self.variables.update(
            error_func=self.error(network_output, prediction),
        )

    def init_methods(self):
        """ Initialize all methods that needed for prediction and
        training procedures.
        """
        super(SupervisedConstructableNetwork, self).init_methods()

        network_input = self.variables.network_input
        network_output = self.variables.network_output

        self.methods.train_epoch = theano.function(
            inputs=[network_input, network_output],
            outputs=self.variables.error_func,
            updates=self.init_train_updates(),
        )
        self.methods.prediction_error = theano.function(
            inputs=[network_input, network_output],
            outputs=self.variables.error_func
        )
