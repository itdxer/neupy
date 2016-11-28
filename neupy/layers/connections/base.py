from itertools import product
from contextlib import contextmanager

from neupy.layers.utils import preformat_layer_shape
from .utils import join, is_sequential
from .graph import LayerGraph


__all__ = ('LayerConnection', 'BaseConnection', 'ParallelConnection')


class BaseConnection(object):
    """
    Base class from chain connections.

    Attributes
    ----------
    connection : LayerConnection or None

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
    def __init__(self):
        self.connection = None
        self.training_state = True
        self.graph = LayerGraph()

        self.input_layers = [self]
        self.output_layers = [self]

    def __gt__(self, other):
        return LayerConnection(self, other)

    def __lt__(self, other):
        return LayerConnection(other, self)

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
            if layers:
                connection = join(*layers)
            else:
                connection = ResidualConnection()

            self.connections.append(connection)
            self.input_layers.extend(connection.input_layers)
            self.output_layers.extend(connection.output_layers)


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

        elif left.connection and left in left.connection.output_layers:
            left = left.connection

        if isinstance(right, (list, tuple)):
            right = ParallelConnection(right)

        elif right.connection and right in right.connection.input_layers:
            right = right.connection

        self.left = left
        self.right = right

        self.left.connection = self
        self.right.connection = self

        layers = product(left.output_layers, right.input_layers)
        for left_output, right_input in layers:
            self.full_graph = make_common_graph(left_output, right_input)

        self.full_graph.connect_layers(left.output_layers, right.input_layers)

        self.input_layers = self.left.input_layers
        self.output_layers = self.right.output_layers

        # Generates subgraph that contains only layers
        # between input and output layers
        self.graph = self.full_graph
        if self.output_layers:
            self.graph = self.graph.subgraph_for_output(self.output_layers)

        if self.input_layers:
            self.graph = self.graph.subgraph_for_output(self.input_layers,
                                                        graph='forward')

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

            - None: In case if connection don't have
              input layers.
        """
        # Cannot save them during initialization step,
        # because input shape can be modified later
        if not self.input_layers:
            return

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

            - None: In case if connection don't have
              output layers.
        """
        # Cannot save them during initialization step,
        # because input shape can be modified later
        if not self.output_layers:
            return

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
