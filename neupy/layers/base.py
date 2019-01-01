import re
import copy
import types
from itertools import chain
from functools import wraps
from abc import abstractmethod
from collections import OrderedDict, defaultdict

import six
import tensorflow as tf

from neupy.core.config import ConfigurableABC
from neupy.exceptions import LayerConnectionError
from neupy.core.properties import (
    Property,
    TypedListProperty,
    ParameterProperty,
)
from neupy.layers.utils import (
    create_shared_parameter,
    make_one_if_possible,
)
from neupy.utils import (
    as_tuple, tensorflow_session,
    initialize_uninitialized_variables,
    class_method_name_scope,
)


__all__ = (
    'BaseGraph', 'LayerGraph',
    'BaseLayer', 'Identity', 'Input',
    'join', 'parallel', 'merge',
)


def filter_graph(dictionary, include_keys):
    """
    Create new list that contains only values
    specified in the ``include_keys`` attribute.

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
    filtered_dict = OrderedDict()

    for key, value in dictionary.items():
        if key in include_keys:
            filtered_dict[key] = [v for v in value if v in include_keys]

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
    >>> is_cyclic({1: [2], 2: [3], 3: [1]})
    True
    >>> is_cyclic({1: [2], 2: [3], 3: [4]})
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


def find_outputs_in_graph(graph):
    outputs = []

    for from_node, to_nodes in graph.items():
        if not to_nodes:
            outputs.append(from_node)

    return outputs


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


def lazy_property(function):
    attr = '_lazy__' + function.__name__

    @property
    @wraps(function)
    def wrapper(self):
        if not hasattr(self, attr):
            setattr(self, attr, function(self))
        return getattr(self, attr)

    return wrapper


class BaseGraph(ConfigurableABC):
    events = []

    def __init__(self, forward_graph=None):
        self.forward_graph = OrderedDict(forward_graph or [])

    @lazy_property
    def backward_graph(self):
        # First we copy all the nodes in order to
        # make sure that order stays the same
        backward = OrderedDict([(node, []) for node in self.forward_graph])

        for to_node, from_nodes in self.forward_graph.items():
            for from_node in from_nodes:
                backward[from_node].append(to_node)

        return backward

    @lazy_property
    def input_layers(self):
        return find_outputs_in_graph(self.backward_graph)

    @lazy_property
    def output_layers(self):
        return find_outputs_in_graph(self.forward_graph)

    @lazy_property
    def inputs(self):
        self.placeholders = []

        for layer in self.input_layers:
            placeholder = tf.placeholder(
                tf.float32,
                shape=as_tuple(None, layer.output_shape),
                name="placeholder/{}".format(layer.name),
            )
            self.placeholders.append(placeholder)

        return make_one_if_possible(self.placeholders)

    @lazy_property
    def outputs(self):
        initialize_uninitialized_variables()
        return self.output(self.inputs)

    def training_outputs(self):
        pass

    @classmethod
    def compare(cls, left, right):
        cls.events.append(('__gt__', join(left, right)))

        graph = LayerGraph()
        previous_operator = None

        for operator, value in reversed(cls.events):
            if operator == previous_operator:
                break

            if operator == '__gt__':
                # It's important to put `value` before graph, because
                # we merge in reverse order and we need to make sure
                # that every new value has higher priority.
                graph = merge(value, graph)

            previous_operator = operator

        return graph

    def __gt__(self, other):
        return self.compare(self, other)

    def __lt__(self, other):
        return self.compare(other, self)

    def __rshift__(self, other):
        return join(self, other)

    def __lshift__(self, other):
        return join(other, self)

    def __bool__(self):
        self.events.append(('__bool__', self))
        return True

    def __or__(self, other):
        return parallel(self, other)

    def __nonzero__(self):
        # Hack for python 2
        return self.__bool__()

    def __contains__(self, entity):
        return entity in self.forward_graph

    def __len__(self):
        return len(self.forward_graph)

    @abstractmethod
    def output(self, inputs):
        raise NotImplementedError()

    @property
    @abstractmethod
    def output_shape(self):
        raise NotImplementedError()

    @abstractmethod
    def get_output_shape(self, input_shape):
        raise NotImplementedError()


class LayerGraph(BaseGraph):
    def reverse(self):
        return self.__class__(self.backward_graph)

    def clean_layer_references(self, layer_references):
        layers = []

        for layer_reference in layer_references:
            if isinstance(layer_reference, six.string_types):
                layer_reference = self.find_layer_by_name(layer_reference)
            layers.append(layer_reference)

        return layers

    def end(self, *output_layers):
        output_layers = self.clean_layer_references(output_layers)

        if not isinstance(output_layers, (list, tuple)):
            output_layers = [output_layers]

        if all(layer not in self.forward_graph for layer in output_layers):
            return self.__class__()

        observed_layers = []
        layers = copy.copy(output_layers)
        backward_graph = self.backward_graph

        while layers:
            current_layer = layers.pop()
            observed_layers.append(current_layer)

            for next_layer in backward_graph[current_layer]:
                if next_layer not in observed_layers:
                    layers.append(next_layer)

        forward_subgraph = filter_graph(self.forward_graph, observed_layers)
        return self.__class__(forward_subgraph)

    def start(self, *input_layers):
        input_layers = self.clean_layer_references(input_layers)

        # Output layers for the reversed graph are
        # input layers for normal graph
        graph_reversed = self.reverse()
        subgraph_reversed = graph_reversed.subgraph_for_output(input_layers)

        # Reverse it to make normal graph
        return subgraph_reversed.reverse()

    @property
    def layers(self):
        return list(self)

    def layer(self, layer_name):
        for layer in self.forward_graph:
            if layer.name == layer_name:
                return layer

        raise NameError("Cannot find layer with name {!r}".format(layer_name))

    def init_variables(self):
        for layer in self:
            layer.init_variables()

    @property
    def input_shape(self):
        return make_one_if_possible(
            [l.output_shape for l in self.input_layers])

    @property
    def output_shape(self):
        return self.get_output_shape(self.input_shape)

    def get_output_shape(self, first_input, *others):
        inputs = as_tuple(first_input, others) if others else first_input
        return self.propagate_forward(inputs, method='get_output_shape')

    def output(self, first_input, *others):
        inputs = as_tuple(first_input, others) if others else first_input
        return self.propagate_forward(inputs, method='output')

    def preformat_inputs(self, inputs):
        if not isinstance(inputs, dict):
            inputs = as_tuple(inputs)
            n_input_layers = len(self.input_layers)
            n_input_vars = len(inputs)

            if n_input_vars != n_input_layers:
                raise ValueError(
                    "Connection has {} input layer(s), but {} inputs was "
                    "provided".format(n_input_layers, n_input_vars))

            inputs = dict(zip(self.input_layers, inputs))

        for layer, input_variable in inputs.items():
            if isinstance(layer, six.string_types):
                layer = self.find_layer_by_name(layer)

            if layer not in self.forward_graph:
                raise ValueError(
                    "The `{}` layer doesn't appear in the network"
                    "".format(layer.name))

        return inputs

    def propagate_forward(self, inputs, method):
        backward_graph = self.backward_graph
        inputs = self.preformat_inputs(inputs)
        inputs = copy.copy(inputs)

        for layer, layer_input in list(inputs.items()):
            layer_method = getattr(layer, method)
            inputs[layer] = layer_method(layer_input)

        for layer in (l for l in self if l not in inputs):
            layer_inputs = make_one_if_possible(
                [inputs[l] for l in backward_graph[layer]])

            layer_method = getattr(layer, method)
            inputs[layer] = layer_method(layer_inputs)

        return make_one_if_possible([inputs[l] for l in self.output_layers])

    def predict(self, *inputs):
        session = tensorflow_session()
        feed_dict = dict(zip(self.inputs, inputs))
        return session.run(self.outputs, feed_dict=feed_dict)

    def is_sequential(self):
        forward_graph_layers = self.forward_graph.values()
        backward_graph_layers = self.backward_graph.values()

        for layers in chain(forward_graph_layers, backward_graph_layers):
            if len(layers) >= 2:
                # One of the layers has multiple input
                # or output connections
                return False

        return True

    def __iter__(self):
        for layer in topological_sort(self.backward_graph):
            yield layer

    def __repr__(self):
        n_layers = len(self)

        if n_layers <= 5 and self.is_sequential():
            return ' > '.join([repr(layer) for layer in self])

        return '{} -> [... {} layers ...] -> {}'.format(
            make_one_if_possible(self.input_shape),
            n_layers,
            make_one_if_possible(self.output_shape))


def merge(left_graph, right_graph, combine=False):
    forward_graph = OrderedDict()

    for key, value in left_graph.forward_graph.items():
        # To make sure that we copied lists inside of the
        # dictionary, but didn't copied values inside of the list
        forward_graph[key] = copy.copy(value)

    for key, values in right_graph.forward_graph.items():
        if key in forward_graph:
            for value in values:
                if value not in forward_graph[key]:
                    forward_graph[key].append(value)
        else:
            forward_graph[key] = copy.copy(values)

    if combine:
        for left_out_layer in left_graph.output_layers:
            for right_in_layer in right_graph.input_layers:
                forward_graph[left_out_layer].append(right_in_layer)

    if is_cyclic(forward_graph):
        raise LayerConnectionError(
            "Cannot create connection between layers, "
            "because it creates cycle in the graph.")

    return LayerGraph(forward_graph)


def parallel(*connections):
    graph = LayerGraph()

    for connection in connections:
        if isinstance(connection, (list, tuple)):
            connection = join(*connection)
        graph = merge(graph, connection)

    return graph


def join(*connections):
    graph = LayerGraph()

    for connection in connections:
        if isinstance(connection, (list, tuple)):
            connection = join(*connection)
        graph = merge(graph, connection, combine=True)

    return graph


def generate_layer_name(layer):
    if not hasattr(generate_layer_name, 'counters'):
        generate_layer_name.counters = defaultdict(int)

    classname = layer.__class__.__name__
    generate_layer_name.counters[classname] += 1
    layer_id = generate_layer_name.counters[classname]

    layer_name = re.sub(r'(?<!^)(?=[A-Z][a-z_])', '-', classname)
    return "{}-{}".format(layer_name.lower(), layer_id)


class BaseLayer(BaseGraph):
    """
    Base class for the layers.

    Parameters
    ----------
    name : str or None
        Layer name. Can be used as a reference to specific layer. When
        value specified as ``None`` than name will be generated from
        the class name. Defaults to ``None``

    Methods
    -------
    variable(value, name, shape=None, trainable=True)
        Initializes variable with specified values.

    get_output_shape(input_shape)
        Computes expected output shape from the layer based on the
        specified input shape.

    output(inputs)
        Propagetes input through the layer.

    Attributes
    ----------
    variable_names : list
        Name of the variables used in the layer.

    variables : dict
        Variable names and their values.
    """
    name = Property(expected_type=six.string_types)

    def __init__(self, name=None):
        # Layer by default gets intialized as a graph with single node in it
        super(BaseLayer, self).__init__(forward_graph=[(self, [])])

        if name is None:
            name = generate_layer_name(layer=self)

        self.updates = []
        self.name = name

        # This decorator ensures that result produced by the
        # `output` method will be marked under layer's name scope.
        self.output = types.MethodType(
            class_method_name_scope(self.output), self)

    @property
    def output_shape(self):
        return self.get_output_shape(None)

    def get_output_shape(self, input_shape):
        return None

    @property
    def variable_names(self):
        names = []

        for name, option in self.options.items():
            if isinstance(option.value, ParameterProperty):
                names.append(name)

        return names

    @property
    def variables(self):
        return {var: getattr(self, var) for var in self.variable_names}

    def variable(self, value, name, shape=None, trainable=True):
        layer_name = 'layer/{layer_name}/{parameter_name}'.format(
            layer_name=self.name,
            parameter_name=name.replace('_', '-'))

        return create_shared_parameter(
            value, layer_name, shape, trainable)

    def __repr__(self):
        classname = self.__class__.__name__
        return '{name}(name={})'.format(self.name, name=classname)

    def __reduce__(self):
        return (self.__class__, self.get_params())


class Identity(BaseLayer):
    def get_output_shape(self, input_shape):
        return input_shape

    def output(self, input_value):
        return input_value


class Input(BaseLayer):
    size = TypedListProperty(element_type=(int, type(None)), allow_none=True)

    def __init__(self, size, name=None):
        self.size = as_tuple(size)
        super(Input, self).__init__(name=name)

    def output(self, inputs):
        return inputs

    def get_output_shape(self, input_shape):
        return input_shape or as_tuple(self.size)

    def __repr__(self):
        return '{name}({size})'.format(
            name=self.__class__.__name__,
            size=make_one_if_possible(self.size))
