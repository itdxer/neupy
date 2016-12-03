from itertools import product
from contextlib import contextmanager

from neupy.layers.utils import preformat_layer_shape
from .utils import join, is_sequential
from .graph import LayerGraph


__all__ = ('LayerConnection', 'BaseConnection', 'ParallelConnection')


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
            self.input_layers.extend(connection.input_layers)
            self.output_layers.extend(connection.output_layers)

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

    def output(self, *input_values):
        """
        Compute outputs per each network in parallel
        connection.

        Parameters
        ----------
        *input_values

        Returns
        -------
        list
        """
        n_inputs = len(input_values)
        n_connections = len(self.connections)

        if n_inputs == 1:
            input_values = list(input_values) * n_connections

        elif n_inputs != n_connections:
            raise ValueError("Expected {} input values for parallel "
                             "connection, got {}".format(n_connections,
                                                         n_inputs))

        outputs = []
        for input_value, connection in zip(input_values, self.connections):
            connection_output = connection.output(input_values)
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

    def output(self, *input_values):
        """
        Propagate input values through all layers in the
        connections and returns output from the final layers.

        Parameters
        ----------
        *input_values
            - Input values can be Theano variables or
              array-like objects

            - Dictionary inputs should have key that
              define input layer and value is a variables
              that needs to be propagated through all layers.

        Returns
        -------
        Theano expression
        """
        subgraph = self.graph
        n_inputs = len(input_values)

        if n_inputs == 1 and not isinstance(input_values[0], dict):
            input_value = input_values[0]
            new_input_values = {}

            for input_layer in self.input_layers:
                new_input_values[input_layer] = input_value

            input_values = [new_input_values]

        return subgraph.propagate_forward(*input_values)

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
        backward_graph = self.graph.backward_graph
        for layer in topological_sort(backward_graph):
            yield layer

    def __repr__(self):
        n_layers = len(self)

        if n_layers > 5 or not is_sequential(self):
            conn = '{} -> [... {} layers ...] -> {}'.format(
                preformat_layer_shape(self.input_shape),
                n_layers,
                preformat_layer_shape(self.output_shape)
            )
        else:
            conn = ' > '.join([repr(layer) for layer in self])

        return conn
