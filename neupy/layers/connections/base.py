import copy
from itertools import product
from contextlib import contextmanager

import six
import theano

from neupy.layers.utils import preformat_layer_shape, create_input_variable
from neupy.utils import as_tuple
from .utils import join, is_sequential
from .graph import LayerGraph


__all__ = ('LayerConnection', 'BaseConnection', 'ParallelConnection')


def create_input_variables(input_layers):
    """
    Create input variables for each input layer
    in the graph.

    Parameters
    ----------
    input_layers : list of layers

    Returns
    -------
    list of Theano variables
    """
    inputs = []

    for input_layer in input_layers:
        variable = create_input_variable(
            input_layer.input_shape,
            name="layer:{}/var:input".format(input_layer.name))
        inputs.append(variable)

    return inputs


def clean_layer_references(graph, layer_references):
    """
    Using list of layers and layer names convert it to
    complete list of layer objects. Function try to find
    layer by specified name in the graph and replate it in the
    layer list to the related object.

    Parameters
    ----------
    graph : LayerGraph instance
        Graph that has information about all layers from
        the ``layer_references`` list.

    layer_references : list, tuple
        List of layer instancens and layer names.

    Returns
    -------
    list of layers
    """
    layers = []

    for layer_reference in layer_references:
        if isinstance(layer_reference, six.string_types):
            layer_reference = graph.find_layer_by_name(layer_reference)

        layers.append(layer_reference)

    return layers


class GlobalConnectionState(dict):
    def __setitem__(self, key, value):
        return super(GlobalConnectionState, self).__setitem__(id(key), value)

    def __getitem__(self, key):
        return super(GlobalConnectionState, self).__getitem__(id(key))

    def __contains__(self, key):
        return super(GlobalConnectionState, self).__contains__(id(key))


class BaseConnection(object):
    """
    Base class from chain connections.

    Attributes
    ----------
    graph : LayerGraph
        Reference to the graph that contains all lations
        between layers specified in the connection.

    training_state : bool
        Defines state of the connection. Defaults to ``True``.

    input_layers : list of layers
        List of connection's input layers.

    output_layers : list of layers
        List of connection's output layers.
    """
    left_states = GlobalConnectionState()
    right_states = GlobalConnectionState()

    def __init__(self):
        self.training_state = True
        self.graph = LayerGraph()

        self.input_layers = [self]
        self.output_layers = [self]

    @classmethod
    def connect(self, left, right):
        """
        Make connection between two objects.
        """
        main_left, main_right = left, right

        if left in self.right_states and left not in self.left_states:
            left = self.right_states[left]

        if right in self.left_states and right not in self.right_states:
            right = self.left_states[right]

        connection = LayerConnection(left, right)

        self.left_states[main_left] = connection
        self.right_states[main_right] = connection

        return connection

    def __gt__(self, other):
        return self.__class__.connect(self, other)

    def __lt__(self, other):
        return self.__class__.connect(other, self)

    def __rshift__(self, other):
        return self.__class__.connect(self, other)

    def __lshift__(self, other):
        return self.__class__.connect(other, self)

    def __iter__(self):
        yield self

    def output(self, input_value):
        """
        Return output base on the input value.

        Parameters
        ----------
        input_value
        """
        raise NotImplementedError

    def initialize(self):
        """
        Initialize connection
        """

    @contextmanager
    def disable_training_state(self):
        """
        Disable connection's training state
        """
        self.training_state = False
        yield
        self.training_state = True

    def compile(self, *inputs):
        """
        Compile Theano function with disabled training state.

        Returns
        -------
        callable object
        """
        if not inputs:
            inputs = create_input_variables(self.input_layers)

        with self.disable_training_state():
            return theano.function(inputs, self.output(*inputs))


def make_common_graph(left_layer, right_layer):
    """
    Makes common graph for two layers that exists
    in different graphs.

    Parameters
    ----------
    left_layer : layer
    right_layer : layer

    Returns
    -------
    LayerGraph instance
        Graph that contains both layers and their connections.
    """
    graph = LayerGraph.merge(left_layer.graph, right_layer.graph)

    for layer in graph.forward_graph.keys():
        layer.graph = graph

    return graph


def topological_sort(graph):
    """
    Repeatedly go through all of the nodes in the graph, moving each of
    the nodes that has all its edges resolved, onto a sequence that
    forms our sorted graph. A node has all of its edges resolved and
    can be moved once all the nodes its edges point to, have been moved
    from the unsorted graph onto the sorted one.

    Parameters
    ----------
    graph : dict
        Dictionary that has graph structure.

    Raises
    ------
    RuntimeError
        If graph has cycles.

    Returns
    -------
    list
        List of nodes sorted in topological order.
    """
    sorted_nodes = []
    graph_unsorted = graph.copy()

    while graph_unsorted:
        acyclic = False

        for node, edges in list(graph_unsorted.items()):
            if all(edge not in graph_unsorted for edge in edges):
                acyclic = True
                del graph_unsorted[node]
                sorted_nodes.append(node)

        if not acyclic:
            raise RuntimeError("A cyclic dependency occurred")

    return sorted_nodes


class ParallelConnection(BaseConnection):
    """
    Connection between separate layer networks in parallel.
    Each network has it's own input and out layers.

    Parameters
    ----------
    connections : list of list or connection
        List of networks.

    Attributes
    ----------
    connections : list of connections
        List of networks.

    input_layers : list of connections
        Combined list of all input layers that each
        parallel connection has.

    output_layers : list of connections
        Combined list of all output layers that each
        parallel connection has.
    """
    def __init__(self, connections):
        from neupy.layers.base import ResidualConnection

        super(ParallelConnection, self).__init__()

        self.connections = []
        self.input_layers = []
        self.output_layers = []

        for layers in connections:
            if isinstance(layers, BaseConnection):
                connection = layers
            elif not layers:
                connection = ResidualConnection()
            else:
                connection = join(*layers)

            self.connections.append(connection)

            for input_layer in connection.input_layers:
                if input_layer not in self.input_layers:
                    self.input_layers.append(input_layer)

            for output_layer in connection.output_layers:
                if output_layer not in self.output_layers:
                    self.output_layers.append(output_layer)

    @property
    def input_shape(self):
        """
        Returns input shape per each network parallel
        connection.
        """
        input_shapes = []
        for connection in self.connections:
            input_shapes.append(connection.input_shape)
        return input_shapes

    @property
    def output_shape(self):
        """
        Returns output shape per each network parallel
        connection.
        """
        output_shapes = []
        for connection in self.connections:
            output_shapes.append(connection.output_shape)
        return output_shapes

    def initialize(self):
        """
        Initialize all connections.
        """
        for connection in self.connections:
            connection.initialize()

    def output(self, first_input, *other_inputs):
        """
        Compute outputs per each network in parallel
        connection.

        Parameters
        ----------
        first_input : Theano variable, array-like, dict
        *other_inputs

        Returns
        -------
        list
        """
        n_inputs = len(other_inputs) + 1  # +1 for first input
        n_connections = len(self.connections)

        if not other_inputs:
            input_values = [first_input] * n_connections

        elif n_inputs == n_connections:
            input_values = as_tuple(first_input, other_inputs)

        else:
            raise ValueError("Expected {} input values for parallel "
                             "connection, got {}"
                             "".format(n_connections, n_inputs))

        outputs = []
        for input_value, connection in zip(input_values, self.connections):
            connection_output = connection.output(input_value)
            outputs.append(connection_output)

        return outputs

    @contextmanager
    def disable_training_state(self):
        """
        Disable training state for all layers in all
        connections.
        """
        for connection in self.connections:
            for layer in connection:
                layer.training_state = False

        yield

        for connection in self.connections:
            for layer in connection:
                layer.training_state = True

    def __iter__(self):
        for connection in self.connections:
            yield connection


class LayerConnection(BaseConnection):
    """
    Make connection between layers.

    Parameters
    ----------
    left : layer, connection or list of connections
    right : layer, connection or list of connections

    Attributes
    ----------
    input_layers : list
        List of input layers.

    input_shape : tuple or list of tuples
        Returns input shape as a tuple in case if network
        has only one input layer and list of tuples otherwise.

    output_layers : list
        List of output layers.

    output_shape : tuple or list of tuples
        Returns output shape as a tuple in case if network
        has only one output layer and list of tuples otherwise.

    layers : list
        Topologicaly sorted list of all layers.

    graph : LayerGraph instance
        Graph that stores relations between layer
        in the network.
    """
    def __init__(self, left, right):
        super(LayerConnection, self).__init__()

        if isinstance(left, (list, tuple)):
            left = ParallelConnection(left)

        if isinstance(right, (list, tuple)):
            right = ParallelConnection(right)

        self.left = left
        self.right = right

        layers = product(left.output_layers, right.input_layers)
        for left_output, right_input in layers:
            self.full_graph = make_common_graph(left_output, right_input)

        self.full_graph.connect_layers(left.output_layers, right.input_layers)

        self.input_layers = self.left.input_layers
        self.output_layers = self.right.output_layers

        # Generates subgraph that contains only connections
        # between specified input and output layers
        self.graph = self.full_graph.subgraph(self.input_layers,
                                              self.output_layers)

    @property
    def input_shape(self):
        """
        Connection's input shape/shapes.

        Returns
        -------
        list of tuples, tuple or None
            - List of tuples: in case if there are more than
              one input layer. Each tuple is a shape of the
              specific input layer.

            - tuple: in case if there is one input layer.
              Tuple object defines input layer's shape.
        """
        # Cannot save them during initialization step,
        # because input shape can be modified later
        if len(self.input_layers) == 1:
            input_layer = self.input_layers[0]
            return input_layer.input_shape

        input_shapes = []
        for input_layer in self.input_layers:
            input_shapes.append(input_layer.input_shape)

        return input_shapes

    @property
    def output_shape(self):
        """
        Connection's output shape/shapes.

        Returns
        -------
        list of tuples, tuple or None
            - List of tuples: in case if there are more than
              one input layer. Each tuple is a shape of the
              specific input layer.

            - tuple: in case if there is one input layer.
              Tuple object defines input layer's shape.
        """
        # Cannot save them during initialization step,
        # because input shape can be modified later
        if len(self.output_layers) == 1:
            output_layer = self.output_layers[0]
            return output_layer.output_shape

        output_shapes = []
        for output_layer in self.output_layers:
            output_shapes.append(output_layer.output_shape)

        return output_shapes

    def initialize(self):
        """
        Initialize all layers in the connection.
        """
        for layer in self:
            layer.initialize()

    def output(self, first_input, *other_inputs):
        """
        Propagate input values through all layers in the
        connections and returns output from the final layers.

        Parameters
        ----------
        first_input : Theano variable, array-like, dict
            - Input values can be Theano variables or
              array-like objects

            - Dictionary inputs should have key that
              define input layer and value is a variables
              that needs to be propagated through all layers.

        *other_inputs
            Suitable in case if we need to set up multiple
            input variables in a sequence.

        Returns
        -------
        Theano expression
        """
        if other_inputs:
            input_values = as_tuple(first_input, other_inputs)
        else:
            input_values = first_input

        if isinstance(input_values, (list, tuple)):
            n_input_layers = len(self.input_layers)
            n_input_vars = len(input_values)

            if n_input_vars != n_input_layers:
                raise ValueError("Connection has {} input layer(s), "
                                 "but {} inputs was provided"
                                 "".format(n_input_layers, n_input_vars))

            # Layers in the self.graph.input_layers and
            # self.input_layers variables can have a different order.
            # Order in the self.input_layers is defined by user
            input_values_as_dict = {}

            for layer, value in zip(self.input_layers, input_values):
                input_values_as_dict[layer] = value

            input_values = input_values_as_dict

        return self.graph.propagate_forward(input_values)

    def start(self, first_layer, *other_layers):
        """
        Create new LayerConnection instance that point to
        different input layers.

        Parameters
        ----------
        first_layer : layer, str
            Layer instance or layer name.

        *other_layers
            Layer instances or layer names.

        Returns
        -------
        connection
        """
        input_layers = as_tuple(first_layer, other_layers)
        input_layers = clean_layer_references(self.graph, input_layers)

        subgraph = self.graph.subgraph_for_input(input_layers)

        new_connection = copy.copy(self)
        new_connection.graph = subgraph
        new_connection.input_layers = subgraph.input_layers

        # don't care about self.left and self.right attributes.
        # remove them to make sure that other function
        # won't use invalid references
        if hasattr(new_connection, 'left'):
            del new_connection.left

        if hasattr(new_connection, 'right'):
            del new_connection.right

        return new_connection

    def end(self, first_layer, *other_layers):
        """
        Create new LayerConnection instance that point to
        different output layers.

        Parameters
        ----------
        first_layer : layer, str
            Layer instance or layer name.

        *other_layers
            Layer instances or layer names.

        Returns
        -------
        connection
        """
        output_layers = as_tuple(first_layer, other_layers)
        output_layers = clean_layer_references(self.graph, output_layers)

        subgraph = self.graph.subgraph_for_output(output_layers)

        new_connection = copy.copy(self)
        new_connection.graph = subgraph
        new_connection.output_layers = subgraph.output_layers

        # don't care about self.left and self.right attributes.
        # remove them to make sure that other function
        # won't use invalid references
        if hasattr(new_connection, 'left'):
            del new_connection.left

        if hasattr(new_connection, 'right'):
            del new_connection.right

        return new_connection

    @property
    def layers(self):
        return list(self)

    def layer(self, layer_name):
        """
        Find layer instance in the network based on the
        specified layer name.

        Parameters
        ----------
        layer_name : str
            Name of the layer.

        Returns
        -------
        layer
        """
        return self.graph.find_layer_by_name(layer_name)

    @contextmanager
    def disable_training_state(self):
        """
        Disable training state for all layers in the
        connection.
        """
        for layer in self:
            layer.training_state = False

        yield

        for layer in self:
            layer.training_state = True

    def __len__(self):
        return len(self.graph.forward_graph)

    def __iter__(self):
        for layer in topological_sort(self.graph.backward_graph):
            yield layer

    def __repr__(self):
        n_layers = len(self)

        if n_layers > 5 or not is_sequential(self):
            return '{} -> [... {} layers ...] -> {}'.format(
                preformat_layer_shape(self.input_shape),
                n_layers,
                preformat_layer_shape(self.output_shape))

        return ' > '.join([repr(layer) for layer in self])
