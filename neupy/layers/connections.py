import copy
import inspect
from itertools import chain
from contextlib import contextmanager
from collections import OrderedDict


__all__ = ('LayerConnection', 'ChainConnection', 'NetworkConnectionError',
           'LayerConnectionError', 'LayerGraph')


class LayerConnectionError(Exception):
    """
    Error class that triggers in case of connection
    issues within layers.
    """


class NetworkConnectionError(Exception):
    """
    Error class that triggers in case of connection
    within layers in the network
    """


def is_feedforward(connection):
    """
    Check whether graph connection is feedforward.

    Parameters
    ----------
    connection : ChainConnection instance

    Returns
    -------
    bool
    """
    graph = connection.graph

    if not graph:
        # Single layer is a feedforward connection
        return True

    f_graph = graph.forward_graph
    b_graph = graph.backward_graph

    for layers in chain(f_graph.values(), b_graph.values()):
        if len(layers) >= 2:
            # One of the layers has multiple input
            # or output connections
            return False

    return True


def filter_dict(dictionary, include_keys):
    """
    Creates new dictionary that contains only some of the
    keys from the original one.

    Parameters
    ----------
    dictionary : dict
        Original dictionary
    include_keys : list or tuple
        Keys that will copied from original dictionary
        into a new one.

    Returns
    -------
    dict
    """
    filtered_dict = {}
    for key, value in dictionary.items():
        if key in include_keys:
            filtered_dict[key] = value
    return filtered_dict


def merge_dicts_with_list(first_dict, second_dict):
    """
    Create new dict that contains all elements from the first
    one and from the second one. Function assumes that all value
    elements are lists. In case if one key appears in both dicts
    function will merge these lists into one.

    Parameters
    ----------
    first_dict : dict
    second_dict : dict

    Returns
    -------
    dict
    """
    common_dict = OrderedDict()

    for key, value in first_dict.items():
        # To make sure that we copied lists inside of the
        # dictionary, but didn't copied values inside of the list
        common_dict[key] = copy.copy(value)

    for key, values in second_dict.items():
        if key not in common_dict:
            common_dict[key] = copy.copy(values)
        else:
            for value in values:
                if value not in common_dict[key]:
                    common_dict[key].append(value)

    return common_dict


def is_cyclic(graph):
    """
    Check if graph has cycles.

    Parameters
    ----------
    graph : dict
        must be represented as a dictionary mapping vertices to
        iterables of neighbouring vertices.

    Returns
    -------
    bool
        Return ``True`` if the directed graph has a cycle.

    Examples
    --------
    >>> is_cyclic({1: (2,), 2: (3,), 3: (1,)})
    True
    >>> is_cyclic({1: (2,), 2: (3,), 3: (4,)})
    False
    """
    path = set()
    visited = set()

    def visit(vertex):
        if vertex in visited:
            return False

        visited.add(vertex)
        path.add(vertex)

        for neighbour in graph.get(vertex, ()):
            if neighbour in path or visit(neighbour):
                return True

        path.remove(vertex)
        return False

    return any(visit(vertex) for vertex in graph)


def does_layer_expect_one_input(layer):
    """
    Check whether layer can except only one input layer.

    Parameters
    ----------
    layer : BaseLayer or LayerConnection instance

    Raises
    ------
    ValueError
        In case if argument is not a layer.

    Retruns
    -------
    bool
        Returns ``True`` if layer can accept onl one input
        layer, ``False`` otherwise.
    """
    if not hasattr(layer, 'output'):
        raise ValueError("Layer `{}` doesn't have "
                         "output method".format(layer))

    if not inspect.ismethod(layer.output):
        raise ValueError("The `output` attribute is not a method.")

    arginfo = inspect.getargspec(layer.output)

    if arginfo.varargs is not None:
        return False

    # In case if layer expects fixed number of input layers
    n_args = len(arginfo.args) - 1  # Ignore `self` argument
    return n_args == 1


class LayerGraph(object):
    """
    Direct Acyclic Graph (DAG) for layer connections.

    Parameters
    ----------
    forward_graph : None or dict
    backward_graph : None or dict
    initialized_graph : None or dict

    Raises
    ------
    LayerConnectionError
        If graph cannot connect layers.
    """
    def __init__(self, forward_graph=None, backward_graph=None,
                 initialized_graph=None):
        if forward_graph is None:
            forward_graph = OrderedDict()

        if backward_graph is None:
            backward_graph = OrderedDict()

        if initialized_graph is None:
            initialized_graph = OrderedDict()

        self.forward_graph = forward_graph
        self.backward_graph = backward_graph
        self.initialized_graph = initialized_graph

    @classmethod
    def merge(cls, left_graph, right_graph):
        """
        Combine two separated graphs into one.

        Parameters
        ----------
        left_graph : LayerGraph instance
        right_graph : LayerGraph instance

        Returns
        -------
        LayerGraph instance
            New graph that contains layers and connections
            from input graphs.
        """
        forward_graph = merge_dicts_with_list(
            left_graph.forward_graph,
            right_graph.forward_graph
        )
        backward_graph = merge_dicts_with_list(
            left_graph.backward_graph,
            right_graph.backward_graph
        )
        initialized_graph = merge_dicts_with_list(
            left_graph.initialized_graph,
            right_graph.initialized_graph
        )
        return cls(forward_graph, backward_graph, initialized_graph)

    def add_layer(self, layer):
        """
        Add new layer into the graph.

        Parameters
        ----------
        layer : hashable object

        Returns
        -------
        bool
            Returns ``False`` if layer has beed already added into
            graph and there is no need to add it again, and
            ``True`` - if layer is a new and was added successfully.
        """
        if layer in self.forward_graph:
            return False

        if layer.input_shape is not None:
            layer.initialize()

        self.forward_graph[layer] = []
        self.backward_graph[layer] = []
        self.initialized_graph[layer] = []

        return True

    def add_connection(self, from_layer, to_layer):
        """
        Add new directional connection between two layers.

        Parameters
        ----------
        from_layer : hashable object
        to_layer : hashable object

        Raises
        ------
        LayerConnectionError
            Raises if it's impossible to connect two layers or
            new connection creates cycles in graph.

        Returns
        -------
        bool
            Returns ``False`` if connection has already been added into
            the graph, and ``True`` if connection was added successfully.
        """
        if from_layer is to_layer:
            raise LayerConnectionError("Cannot connect layer `{}` "
                                       "to itself".format(from_layer))

        self.add_layer(from_layer)
        self.add_layer(to_layer)

        expect_one_input_layer = does_layer_expect_one_input(to_layer)
        forward_connections = self.forward_graph[from_layer]
        backward_connections = self.backward_graph[to_layer]

        if to_layer in forward_connections:
            # Layers have been already connected
            return False

        if expect_one_input_layer and backward_connections:
            raise LayerConnectionError(
                "Cannot connect `{from_layer}` to the `{to_layer}`. "
                "Layer `{to_layer}` expectes input only from one "
                "layer and it has been alredy connected with "
                "`{to_layer_connection}`.".format(
                    from_layer=from_layer,
                    to_layer=to_layer,
                    to_layer_connection=backward_connections[0]
                )
            )

        forward_connections.append(to_layer)

        if is_cyclic(self.forward_graph):
            # Rollback changes in case if user cathes exception
            self.forward_graph[from_layer].pop()
            raise LayerConnectionError("Graph cannot have cycles")

        backward_connections.append(from_layer)
        return True

    def connect_layers(self, from_layer, to_layer):
        """
        Connect two layers together and update other layers
        in the graph.

        Parameters
        ----------
        from_layer : hashable object
        to_layer : hashable object

        Raises
        ------
        LayerConnectionError
            Raises if cannot graph cannot connect two layers.

        Returns
        -------
        bool
            Returns ``False`` if connection has already been added into
            the graph, and ``True`` if connection was added successfully.
        """
        connection_added = self.add_connection(from_layer, to_layer)

        if not connection_added:
            return False

        if from_layer.input_shape is None:
            return True

        # Layer has an input shape which means that we can
        # propagate this information through the graph and
        # set up input shape for layers that don't have it.
        layers = [from_layer]

        initialized_graph = self.initialized_graph
        forward_graph = self.forward_graph

        while layers:
            current_layer = layers.pop()
            next_layers = forward_graph[current_layer]

            current_layer_init_rel = initialized_graph[current_layer]

            for next_layer in next_layers:
                if next_layer in current_layer_init_rel:
                    continue

                in_shape = next_layer.input_shape
                out_shape = current_layer.output_shape
                one_input_layer = does_layer_expect_one_input(next_layer)

                if not in_shape or not one_input_layer:
                    next_layer.input_shape = out_shape
                    current_layer_init_rel.append(next_layer)
                    next_layer.initialize()

                elif one_input_layer and in_shape != out_shape:
                    raise LayerConnectionError(
                        "Cannot connect `{}` to the `{}`. Output shape "
                        "from one layer is equal to {} and input shape "
                        "to the next one is equal to {}".format(
                            current_layer, next_layer,
                            out_shape, in_shape,
                        )
                    )

            layers.extend(next_layers)

        return True

    def subgraph_for_output(self, layer):
        layers = [layer]
        observed_layers = [layer]

        if layer not in self.forward_graph:
            return LayerGraph()

        while layers:
            current_layer = layers.pop()
            next_layers = self.backward_graph[current_layer]

            layers.extend(next_layers)
            observed_layers.extend(next_layers)

        forward_subgraph = filter_dict(self.forward_graph,
                                       observed_layers)
        backward_subgraph = filter_dict(self.backward_graph,
                                        observed_layers)
        initialized_subgraph = filter_dict(self.initialized_graph,
                                           observed_layers)

        # Remove old relations to the other layers.
        # Output layer cannot point to some other layers.
        forward_subgraph[layer] = []

        return LayerGraph(forward_subgraph, backward_subgraph,
                          initialized_subgraph)

    @property
    def input_layers(self):
        """
        List of input layers.

        Raises
        ------
        LayerConnectionError
            If graph doesn't have input layers.

        Returns
        -------
        list
            List of input layers.
        """
        input_layers = []
        for layer, next_layers in self.backward_graph.items():
            # TODO: I should check whether it's always useful
            # to have only an input layers that have specified
            # input shape
            if not next_layers and layer.input_shape:
                input_layers.append(layer)

        if not input_layers:
            raise LayerConnectionError("Graph doesn't have input layers")

        return input_layers

    @property
    def output_layers(self):
        """
        List of output layers.

        Raises
        ------
        LayerConnectionError
            If graph doesn't have output layers.

        Returns
        -------
        list
            List of output layers.
        """
        output_layers = []
        for layer, next_layers in self.forward_graph.items():
            if not next_layers:
                output_layers.append(layer)

        if not output_layers:
            raise LayerConnectionError("Graph doesn't have output layers")

        return output_layers

    def propagate_forward(self, input_):
        """
        Propagates input variable through the directed acyclic
        graph and returns output from the final layers.

        Parameters
        ----------
        input_ : array-like, Theano variable or dict
            If input has array or Theano variable type than it will
            be used as a direct input for input layer/layers. The
            dict type input should has a specific structure. Each
            key of the dict is a layer and each value is array or
            Theano variable. Dict defines input values for specific
            layers. In the dict input layer is not necessary should
            be an instance of the ``layers.Input`` class. It can be
            any layer from the graph.

        Returns
        -------
        object
            Output from the final layers.
        """
        outputs = {}

        if isinstance(input_, dict):
            for layer, input_variable in input_.items():
                if layer not in self.forward_graph:
                    raise ValueError("The `{}` layer doesn't appear "
                                     "in this graph".format(layer))

                outputs[layer] = layer.output(input_variable)

        else:
            for input_layer in self.input_layers:
                outputs[input_layer] = input_layer.output(input_)

        def output_from_layer(layer):
            if layer in outputs:
                return outputs[layer]

            input_layers = self.backward_graph[layer]
            inputs = []
            for input_layer in input_layers:
                if input_layer in outputs:
                    res = outputs[input_layer]
                else:
                    res = output_from_layer(input_layer)
                    outputs[input_layer] = res

                inputs.append(res)

            return layer.output(*inputs)

        results = []
        for output_layer in self.output_layers:
            results.append(output_from_layer(output_layer))

        if len(results) == 1:
            results = results[0]

        return results


class ChainConnection(object):
    """
    Base class from chain connections.
    """
    graph = None

    def __init__(self):
        self.connection = None
        self.training_state = True

    def __gt__(self, other):
        return LayerConnection(self, other)

    def __lt__(self, other):
        return LayerConnection(other, self)

    def output(self, input_value):
        raise NotImplementedError

    def initialize(self):
        pass

    @contextmanager
    def disable_training_state(self):
        self.training_state = False
        yield
        self.training_state = True


def make_common_graph(left_layer, right_layer):
    """
    Makes common graph for two layers that exists
    in different graphs.

    Parameters
    ----------
    left_layer : BaseLayer instance
    right_layer : BaseLayer instance

    Returns
    -------
    LayerGraph instance
        Graph that contains both layers and their connections.
    """
    left_graph = left_layer.graph
    right_graph = right_layer.graph

    if left_graph is None:
        left_graph = LayerGraph()

    if right_graph is None:
        right_graph = LayerGraph()

    graph = LayerGraph.merge(left_graph, right_graph)

    for layer in graph.forward_graph.keys():
        layer.graph = graph

    left_layer.graph = graph
    right_layer.graph = graph

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
            for edge in edges:
                if edge in graph_unsorted:
                    break
            else:
                acyclic = True
                del graph_unsorted[node]
                sorted_nodes.append(node)

        if not acyclic:
            raise RuntimeError("A cyclic dependency occurred")

    return sorted_nodes


class LayerConnection(ChainConnection):
    """
    Connect to layers or connections together.

    Parameters
    ----------
    left : ChainConnection instance
    right : ChainConnection instance
    """
    def __init__(self, left, right):
        super(LayerConnection, self).__init__()

        if left.connection and left.connection.output_layer is left:
            self.left = left.connection
        else:
            self.left = left

        if right.connection and right.connection.input_layer is right:
            self.right = right.connection
        else:
            self.right = right

        self.layers = []

        if isinstance(self.left, LayerConnection):
            self.layers = copy.copy(self.left.layers)
            self.left_layer = self.layers[-1]
        else:
            self.left_layer = self.left
            self.layers = [self.left]

        if isinstance(self.right, LayerConnection):
            right_layers = self.right.layers
            self.right_layer = right_layers[0]
            self.layers.extend(right_layers)
        else:
            self.right_layer = self.right
            self.layers.append(self.right)

        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]
        self.input_shape = self.input_layer.input_shape
        self.output_shape = self.output_layer.output_shape

        self.left.connection = self
        self.right.connection = self

        self.graph = make_common_graph(self.left_layer, self.right_layer)
        self.graph.connect_layers(self.left_layer, self.right_layer)

    def initialize(self):
        for layer in self:
            layer.initialize()

    def output(self, *input_values):
        subgraph = self.graph.subgraph_for_output(self.output_layer)
        return subgraph.propagate_forward(*input_values)

    @contextmanager
    def disable_training_state(self):
        for layer in self:
            layer.training_state = False

        yield

        for layer in self:
            layer.training_state = True

    def __len__(self):
        layers = list(iter(self))
        return len(layers)

    def __iter__(self):
        subgraph = self.graph.subgraph_for_output(self.output_layer)
        for layer in topological_sort(subgraph.backward_graph):
            yield layer

    def __repr__(self):
        layers_reprs = map(repr, self)
        return ' > '.join(layers_reprs)
