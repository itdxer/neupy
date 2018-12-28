import copy
import types
from functools import wraps
from itertools import chain
from contextlib import contextmanager

import six
import tensorflow as tf

from neupy.layers.utils import preformat_layer_shape, find_variables
from neupy.utils import (
    as_tuple, tensorflow_session,
    initialize_uninitialized_variables,
)
from .graph import LayerGraph


__all__ = ('ExecutableGraph', 'BaseConnection', 'parallel', 'join')


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


def is_sequential(connection):
    """
    Check whether graph connection is a sequence.

    Parameters
    ----------
    connection : connection

    Returns
    -------
    bool
    """
    forward_graph_layers = connection.graph.forward_graph.values()
    backward_graph_layers = connection.graph.backward_graph.values()

    for layers in chain(forward_graph_layers, backward_graph_layers):
        if len(layers) >= 2:
            # One of the layers has multiple input
            # or output connections
            return False

    return True


def check_initialization(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(*args, **kwargs)
        self.initialized = True
        return result
    return wrapper


def create_input_variables(input_layers):
    """
    Create input variables for each input layer
    in the graph.

    Parameters
    ----------
    input_layers : list of layers

    Returns
    -------
    list of Tensorflow variables
    """
    inputs = []

    for input_layer in input_layers:
        variable = tf.placeholder(
            tf.float32,
            shape=as_tuple(None, input_layer.output_shape),
            name="network-input/to-layer-{}".format(input_layer.name),
        )
        inputs.append(variable)

    return inputs


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
    events = []
    computation_cache = {}

    def __init__(self):
        self.initialized = False
        self.training_state = True
        self.graph = LayerGraph()

        # Make sure that we save information when connection was
        # initialized. It will work even if method was reinitialized
        self.initialize = types.MethodType(
            check_initialization(self.initialize), self)

    @property
    def input_layers(self):
        return self.graph.input_layers

    @property
    def output_layers(self):
        return self.graph.output_layers

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
        input_shapes = []
        for input_layer in self.input_layers:
            input_shapes.append(input_layer.input_shape)
        return input_shapes[0] if len(input_shapes) == 1 else input_shapes

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
        output_shapes = []
        for output_layer in self.output_layers:
            output_shapes.append(output_layer.output_shape)
        return output_shapes[0] if len(output_shapes) == 1 else output_shapes

    def compare(self, left, right):
        self.events.append(('__gt__', left, right, join(left, right)))

        subgraph = LayerGraph()
        previous_operator = None

        for operation in reversed(self.events):
            operator = operation[0]

            if operator == previous_operator:
                break

            if operator == '__gt__':
                _, left_operation, right_operation, cached_conn = operation
                subgraph = LayerGraph.merge(subgraph, cached_conn.graph)

            previous_operator = operator

        return ExecutableGraph(subgraph)

    def __gt__(self, other):
        return self.compare(self, other)

    def __lt__(self, other):
        return self.compare(other, self)

    def __rshift__(self, other):
        return self.compare(self, other)

    def __lshift__(self, other):
        return self.compare(other, self)

    def __bool__(self):
        self.events.append(('__bool__', self))
        return True

    def __nonzero__(self):
        # Hack for python 2
        return self.__bool__()

    def __iter__(self):
        yield self

    def output(self, input_value):
        """
        Return output base on the input value.

        Parameters
        ----------
        input_value
        """
        raise NotImplementedError()

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

    def predict(self, *inputs):
        """
        Using current tensorflow session this method propagates
        input throught the network and returns output from it.
        """
        session = tensorflow_session()
        # We cache it in order to avoid graph creation
        # every time user calls prediction.
        cache_key = (session, id(self))

        if cache_key not in self.computation_cache:
            # It's important to initialize parameters when for cases when
            # prediction requested for the network that wasn't trained or
            # loaded from the storage.
            variables = find_variables(self.layers)
            initialize_uninitialized_variables(variables)

            input_variables = create_input_variables(self.input_layers)

            with self.disable_training_state():
                self.computation_cache[cache_key] = {
                    'inputs': input_variables,
                    'outputs': self.output(*input_variables),
                }

        graph = self.computation_cache[cache_key]
        feed_dict = dict(zip(graph['inputs'], inputs))

        return session.run(graph['outputs'], feed_dict=feed_dict)


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


def parallel(*connections):
    from neupy.layers.base import Identity

    graph = LayerGraph()

    for connection in connections:
        if isinstance(connection, (list, tuple)):
            connection = join(*connection) if connection else Identity()
        graph = LayerGraph.merge(graph, connection.graph)

    return ExecutableGraph(graph)


def join(*connections):
    """
    Connect two layers.

    Parameters
    ----------
    *connections : layers or connections

    Returns
    -------
    connection
        Layers connected in a sequence.

    Examples
    --------
    >>> from neupy import layers
    >>> conn = layers.join(
    ...     layers.Input(784),
    ...     layers.Relu(500),
    ...     layers.Relu(300),
    ...     layers.Softmax(10),
    ... )
    """
    graph = LayerGraph()

    for connection in connections:
        if isinstance(connection, (list, tuple)):
            connection = parallel(*connection)

        merged_graph = LayerGraph.merge(graph, connection.graph)

        if merged_graph.output_layers:
            merged_graph.connect_layers(
                graph.output_layers,
                connection.graph.input_layers)

        graph = merged_graph

    return ExecutableGraph(graph)


class ExecutableGraph(BaseConnection):
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
    def __init__(self, graph):
        super(ExecutableGraph, self).__init__()
        self.graph = graph

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
        first_input : Tensorfow variable, array-like, dict
            - Input values can be Tensorfow variables or
              array-like objects

            - Dictionary inputs should have key that
              define input layer and value is a variables
              that needs to be propagated through all layers.

        *other_inputs
            Suitable in case if we need to set up multiple
            input variables in a sequence.

        Returns
        -------
        Tensorfow expression
        """
        if other_inputs:
            input_values = as_tuple(first_input, other_inputs)
        else:
            input_values = first_input

        if isinstance(input_values, (list, tuple)):
            n_input_layers = len(self.input_layers)
            n_input_vars = len(input_values)

            if n_input_vars != n_input_layers:
                raise ValueError(
                    "Connection has {} input layer(s), but {} inputs was "
                    "provided".format(n_input_layers, n_input_vars))

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
        return ExecutableGraph(subgraph)

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
        return ExecutableGraph(subgraph)

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

        if n_layers <= 5 and is_sequential(self):
            return ' > '.join([repr(layer) for layer in self])

        return '{} -> [... {} layers ...] -> {}'.format(
            preformat_layer_shape(self.input_shape),
            n_layers,
            preformat_layer_shape(self.output_shape))
