import inspect
from contextlib import contextmanager
from collections import defaultdict


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


def filter_dict(dictionary, valid_keys):
    filtered_dict = {}
    for key, value in dictionary.items():
        if key in valid_keys:
            filtered_dict[key] = value
    return filtered_dict


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
    def __init__(self, forward_graph=None, backward_graph=None,
                 initialized_graph=None):
        if forward_graph is None:
            forward_graph = defaultdict(list)

        if backward_graph is None:
            backward_graph = defaultdict(list)

        if initialized_graph is None:
            initialized_graph = defaultdict(list)

        self.forward_graph = forward_graph
        self.backward_graph = backward_graph
        self.initialized_graph = initialized_graph

    @classmethod
    def merge(cls, left_graph, right_graph):
        left_forward_graph = left_graph.forward_graph.copy()
        left_backward_graph = left_graph.backward_graph.copy()

        forward_graph = left_forward_graph.update(right_graph.forward_graph)
        backward_graph = left_backward_graph.update(right_graph.backward_graph)

        return cls(forward_graph, backward_graph)

    def add_vertex(self, vertex):
        if vertex not in self.forward_graph:
            self.forward_graph[vertex] = []
            self.backward_graph[vertex] = []
            return True
        return False

    def add_edge(self, from_vertex, to_vertex):
        if from_vertex is to_vertex:
            raise LayerConnectionError("Cannot connect layer `{}` "
                                       "to itself".format(from_vertex))

        self.add_vertex(from_vertex)
        self.add_vertex(to_vertex)

        expect_one_input_layer = does_layer_expect_one_input(to_vertex)
        forward_connections = self.forward_graph[from_vertex]
        backward_connections = self.backward_graph[to_vertex]

        if to_vertex in forward_connections:
            # Layers have been already connected
            return False

        if expect_one_input_layer and backward_connections:
            raise LayerConnectionError(
                "Cannot connect `{from_layer}` to the `{to_layer}`. "
                "Layer `{to_layer}` expectes input only from one "
                "layer and it has been alredy connected with "
                "`{to_layer_connection}`.".format(
                    from_layer=from_vertex,
                    to_layer=to_vertex,
                    to_layer_connection=backward_connections[0]
                )
            )
        forward_connections.append(to_vertex)

        if is_cyclic(self.forward_graph):
            # Rollback changes in case if user cathes exception
            self.forward_graph[from_vertex].pop()
            raise LayerConnectionError("Graph cannot have cycles")

        backward_connections.append(from_vertex)
        return True

    def connect_layers(self, from_vertex, to_vertex):
        edge_added = self.add_edge(from_vertex, to_vertex)

        if from_vertex.input_shape is None or not edge_added:
            return

        # Layer has an input shape which means that we can
        # propagate this information through the graph and
        # set up input shape for layers that don't have it.
        vertecies = [from_vertex]

        while vertecies:
            current_vertex = vertecies.pop()
            next_vertecies = self.forward_graph[current_vertex]

            for next_vertex in next_vertecies:
                if next_vertex in self.initialized_graph[current_vertex]:
                    continue

                in_shape = next_vertex.input_shape
                out_shape = current_vertex.output_shape
                one_input_layer = does_layer_expect_one_input(next_vertex)

                if not in_shape or not one_input_layer:
                    next_vertex.input_shape = out_shape
                    self.initialized_graph[current_vertex].append(next_vertex)
                    next_vertex.initialize()

                elif one_input_layer and in_shape != out_shape:
                    raise LayerConnectionError(
                        "Cannot connect `{}` to the `{}`. Output shape "
                        "from one layer is equal to {} and input shape "
                        "to the next one is equal to {}".format(
                            current_vertex, next_vertex,
                            out_shape, in_shape,
                        )
                    )

            vertecies.extend(next_vertecies)

    def subgraph_for_output(self, vertex):
        vertecies = [vertex]
        observed_vertecies = [vertex]

        while vertecies:
            current_vertex = vertecies.pop()
            next_vertecies = self.backward_graph[current_vertex]

            vertecies.extend(next_vertecies)
            observed_vertecies.extend(next_vertecies)

        forward_subgraph = filter_dict(self.forward_graph,
                                       observed_vertecies)
        backward_subgraph = filter_dict(self.backward_graph,
                                        observed_vertecies)

        return LayerGraph(forward_subgraph, backward_subgraph)

    @property
    def input_vertecies(self):
        input_vertecies = []
        for vertex, next_vertecies in self.backward_graph.items():
            if not next_vertecies:
                input_vertecies.append(vertex)

        if not input_vertecies:
            raise LayerConnectionError("Graph doesn't have input vertecies")

        return input_vertecies

    @property
    def output_vertecies(self):
        output_vertecies = []
        for vertex, next_vertecies in self.forward_graph.items():
            if not next_vertecies:
                output_vertecies.append(vertex)

        if not output_vertecies:
            raise LayerConnectionError("Graph doesn't have output vertecies")

        return output_vertecies

    def propagate_forward(self, input_):
        outputs = {}
        for input_vertex in self.input_vertecies:
            outputs[input_vertex] = input_vertex.output(input_)

        def output_from_vertex(vertex):
            input_vertecies = self.backward_graph[vertex]
            inputs = []
            for input_vertex in input_vertecies:
                if input_vertex in outputs:
                    res = outputs[input_vertex]
                else:
                    res = output_from_vertex(input_vertex)
                    outputs[input_vertex] = res

                inputs.append(res)

            return vertex.output(*inputs)

        results = []
        for output_vertex in self.output_vertecies:
            results.append(output_from_vertex(output_vertex))

        if len(results) == 1:
            results = results[0]

        return results


class ChainConnection(object):
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
    left_graph = left_layer.graph
    right_graph = right_layer.graph

    if left_graph is None and right_graph is None:
        graph = LayerGraph()

    elif left_graph is not None and right_graph is None:
        graph = left_graph

    elif left_graph is None and right_graph is not None:
        graph = right_graph

    elif left_graph is not None and right_graph is not None:
        if left_graph is right_graph:
            graph = left_graph
        else:
            graph = LayerGraph.merge(left_layer.graph, right_layer.graph)

    for layer in graph.forward_graph.keys():
        layer.graph = graph

    left_layer.graph = graph
    right_layer.graph = graph

    return graph


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
            self.layers = self.left.layers
            self.left_layer = self.layers[-1]
        else:
            self.left_layer = self.left
            self.layers = [self.left]

        if isinstance(self.right, LayerConnection):
            temp = self.right.layers
            self.right_layer = temp[0]
            self.layers.extend(temp)
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
        for layer in self.layers:
            layer.initialize()

    def output(self, *input_values):
        subgraph = self.graph.subgraph_for_output(self.output_layer)
        return subgraph.propagate_forward(*input_values)

    def __len__(self):
        layers = list(iter(self))
        return len(layers)

    def __iter__(self):
        if isinstance(self.left, LayerConnection):
            for conn in self.left:
                yield conn
        else:
            yield self.left

        if isinstance(self.right, LayerConnection):
            for conn in self.right:
                yield conn
        else:
            yield self.right

    def __repr__(self):
        layers_reprs = map(repr, self)
        return ' > '.join(layers_reprs)
