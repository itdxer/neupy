import sys
import copy
import tempfile
from itertools import chain
from functools import wraps
from abc import abstractmethod
from collections import OrderedDict

import six
import graphviz
import numpy as np
import tensorflow as tf

from neupy.core.config import ConfigurableABC, DumpableObject
from neupy.exceptions import LayerConnectionError
from neupy.utils import as_tuple, tf_utils, iters


__all__ = (
    'BaseGraph', 'LayerGraph',
    'join', 'parallel', 'merge', 'repeat',
)


def make_one_if_possible(shape):
    """
    Format layer's input or output shape.

    Parameters
    ----------
    shape : int or tuple

    Returns
    -------
    int or tuple
    """
    if isinstance(shape, (tuple, list)) and len(shape) == 1:
        return shape[0]
    return shape


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
    if not graph:
        return []

    if is_cyclic(graph):
        raise RuntimeError(
            "Cannot apply topological sort to the graphs with cycles")

    sorted_nodes = []
    graph_unsorted = graph.copy()

    while graph_unsorted:
        for node, edges in list(graph_unsorted.items()):
            if all(edge not in graph_unsorted for edge in edges):
                del graph_unsorted[node]
                sorted_nodes.append(node)

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


class BaseGraph(ConfigurableABC, DumpableObject):
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
        placeholders = []

        for layer in self.input_layers:
            placeholder = tf.placeholder(
                tf.float32,
                shape=tf_utils.shape_to_tuple(layer.input_shape),
                name="placeholder/input/{}".format(layer.name),
            )
            placeholders.append(placeholder)

        return make_one_if_possible(placeholders)

    @lazy_property
    def targets(self):
        placeholders = []

        for layer in self.output_layers:
            placeholder = tf.placeholder(
                tf.float32,
                shape=tf_utils.shape_to_tuple(layer.output_shape),
                name="placeholder/target/{}".format(layer.name),
            )
            placeholders.append(placeholder)

        return make_one_if_possible(placeholders)

    @lazy_property
    def outputs(self):
        networks_output = self.output(*as_tuple(self.inputs))
        tf_utils.initialize_uninitialized_variables()
        return networks_output

    @lazy_property
    def training_outputs(self):
        networks_output = self.output(*as_tuple(self.inputs), training=True)
        tf_utils.initialize_uninitialized_variables()
        return networks_output

    def __gt__(self, other):
        left, right = self, other
        self.events.append(('__gt__', join(left, right)))

        graph = LayerGraph()
        previous_operator = None

        for operator, value in reversed(self.events):
            if operator == previous_operator:
                break

            if operator == '__gt__':
                # It's important to put `value` before graph, because
                # we merge in reverse order and we need to make sure
                # that every new value has higher priority.
                graph = merge(value, graph)

            previous_operator = operator

        return graph

    def __bool__(self):
        self.events.append(('__bool__', self))
        return True

    def __nonzero__(self):
        return self.__bool__()  # Hack for python 2

    def __rshift__(self, other):
        return join(self, other)

    def __irshift__(self, other):
        return self.__rshift__(other)

    def __or__(self, other):
        return parallel(self, other)

    def __ior__(self, other):
        return self.__or__(other)

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
    def __init__(self, forward_graph=None):
        super(LayerGraph, self).__init__(forward_graph)

        # This allows to run simple check that ensures that
        # created graph have defined layer shape
        self.output_shape

    def clean_layer_references(self, layer_references):
        layers = []

        for layer_reference in layer_references:
            if isinstance(layer_reference, six.string_types):
                layer_reference = self.layer(layer_reference)
            layers.append(layer_reference)

        return layers

    def slice(self, directed_graph, layers):
        layers = self.clean_layer_references(layers)
        forward_graph = self.forward_graph

        if all(layer not in forward_graph for layer in layers):
            unused_layer = next(l for l in layers if l not in forward_graph)
            raise ValueError(
                "Layer `{}` is not used in the graph. Graph: {}, "
                "Layer: {}".format(unused_layer.name, self, unused_layer))

        observed_layers = []
        layers = copy.copy(layers)

        while layers:
            current_layer = layers.pop()
            observed_layers.append(current_layer)

            for next_layer in directed_graph[current_layer]:
                if next_layer not in observed_layers:
                    layers.append(next_layer)

        forward_subgraph = filter_graph(forward_graph, observed_layers)
        return self.__class__(forward_subgraph)

    def end(self, *output_layers):
        return self.slice(self.backward_graph, output_layers)

    def start(self, *input_layers):
        return self.slice(self.forward_graph, input_layers)

    @lazy_property
    def layers(self):
        return list(self)

    def layer(self, layer_name):
        if not isinstance(layer_name, six.string_types):
            raise ValueError(
                "Layer name expected to be a string, "
                "got value {}".format(layer_name))

        layers = []

        for layer in self.forward_graph:
            if layer.name == layer_name:
                layers.append(layer)

        if not layers:
            raise NameError(
                "Cannot find layer with name {!r}".format(layer_name))

        if len(layers) >= 2:
            raise NameError(
                "Ambiguous layer name `{}`. Network has {} "
                "layers with the same name. Layers: {}".format(
                    layer_name, len(layers), layers))

        return layers[0]

    @lazy_property
    def input_shapes(self):
        return [tf.TensorShape(l.input_shape) for l in self.input_layers]

    @lazy_property
    def input_shape(self):
        return make_one_if_possible(self.input_shapes)

    @lazy_property
    def output_shape(self):
        return self.get_output_shape(*self.input_shapes)

    @lazy_property
    def output_shapes_per_layer(self):
        return self.propagate_forward(
            copy.deepcopy(self.input_shapes),
            method='get_output_shape')

    def get_output_shape(self, *inputs):
        outputs = self.propagate_forward(
            copy.deepcopy(inputs),
            method='get_output_shape',
        )
        return make_one_if_possible(
            [outputs[l] for l in self.output_layers])

    def create_variables(self):
        output_shapes = self.output_shapes_per_layer
        backward_graph = self.backward_graph

        for layer in self:
            input_shapes = [layer.input_shape]
            from_layers = backward_graph[layer]

            if layer.frozen:
                continue

            if from_layers:
                input_shapes = [output_shapes[l] for l in from_layers]

            layer.create_variables(*input_shapes)
            layer.frozen = True

    def output(self, *inputs, **kwargs):
        self.create_variables()
        outputs = self.propagate_forward(inputs, method='output', **kwargs)
        return make_one_if_possible([outputs[l] for l in self.output_layers])

    def preformat_inputs(self, inputs):
        if len(inputs) == 1 and isinstance(inputs[0], dict):
            inputs = inputs[0]

        if not isinstance(inputs, dict):
            n_input_layers = len(self.input_layers)
            n_input_vars = len(inputs)

            if n_input_vars != n_input_layers:
                raise ValueError(
                    "Connection has {} input layer(s), but {} inputs was "
                    "provided".format(n_input_layers, n_input_vars))

            inputs = dict(zip(self.input_layers, inputs))

        prepared_inputs = {}
        for layer, input_variable in inputs.items():
            if isinstance(layer, six.string_types):
                layer = self.layer(layer)

            if layer not in self.forward_graph:
                raise ValueError(
                    "The `{}` layer doesn't appear in the network"
                    "".format(layer.name))

            if layer not in self.input_layers:
                raise ValueError(
                    "`{}` is not an input layer in the network"
                    "".format(layer.name))

            prepared_inputs[layer] = input_variable

        return prepared_inputs

    def pass_through_the_layer(self, layer, method, *args, **kwargs):
        layer_method = getattr(layer, method)

        try:
            return layer_method(*args, **kwargs)
        except Exception as exception:
            modified_exception = exception.__class__(
                "{original_message}. Exception occurred while propagating "
                "data through the method `{method}`. Layer: {layer!r}".format(
                    original_message=str(exception).strip('.'),
                    method=method, layer=layer
                )
            )

            if hasattr(sys, 'last_traceback') and six.PY3:
                modified_exception = modified_exception.with_traceback(
                    sys.last_traceback)

            raise modified_exception

    def propagate_forward(self, inputs, method, **kwargs):
        backward_graph = self.backward_graph
        inputs = self.preformat_inputs(inputs)
        outputs = copy.copy(inputs)

        for layer, layer_input in inputs.items():
            outputs[layer] = self.pass_through_the_layer(
                layer, method, layer_input, **kwargs)

        for layer in (l for l in self if l not in inputs):
            layer_inputs = [outputs[l] for l in backward_graph[layer]]
            outputs[layer] = self.pass_through_the_layer(
                layer, method, *layer_inputs, **kwargs)

        return outputs

    @property
    def variables(self):
        self.create_variables()

        variables = OrderedDict()
        observed_variables = []

        for layer in self:
            for name, value in layer.variables.items():
                if value not in observed_variables:
                    observed_variables.append(value)
                    variables[(layer, name)] = value

        return variables

    @property
    def n_parameters(self):
        n_parameters = 0

        for variable in self.variables.values():
            n_parameters += variable.shape.num_elements()

        return n_parameters

    def predict(self, *inputs, **kwargs):
        session = tf_utils.tensorflow_session()

        batch_size = kwargs.pop('batch_size', None)
        verbose = kwargs.pop('verbose', True)

        # We require do to this check for python 2 compatibility
        if kwargs:
            raise TypeError("Unknown arguments: {}".format(kwargs))

        def single_batch_predict(*inputs):
            feed_dict = dict(zip(as_tuple(self.inputs), inputs))
            return session.run(self.outputs, feed_dict=feed_dict)

        outputs = iters.apply_batches(
            function=single_batch_predict,
            inputs=inputs,
            batch_size=batch_size,
            show_progressbar=verbose,
        )
        return np.concatenate(outputs, axis=0)

    def is_sequential(self):
        if len(self.input_layers) > 1 or len(self.output_layers) > 1:
            return False

        forward_graph_layers = self.forward_graph.values()
        backward_graph_layers = self.backward_graph.values()

        for layers in chain(forward_graph_layers, backward_graph_layers):
            if len(layers) >= 2:
                # One of the layers has multiple input
                # or output networks
                return False

        return True

    def layer_names_only(self):
        prepared_graph = OrderedDict()

        for from_layer, to_layers in self.forward_graph.items():
            prepared_graph[from_layer.name] = [l.name for l in to_layers]

        return list(prepared_graph.items())

    def show(self, filepath=None):
        """
        Generates visual representation of the network. Method will create
        PDF that will be rendered and opened automatically. Graph will
        be stored in the ``filepath`` when it's specified.
        """
        if filepath is None:
            filepath = tempfile.mktemp()

        def layer_uid(layer):
            return str(id(layer))

        digraph = graphviz.Digraph()
        shapes_per_layer = self.output_shapes_per_layer

        for layer in self.forward_graph.keys():
            digraph.node(layer_uid(layer), str(layer.name))

        output_id = 1
        for from_layer, to_layers in self.forward_graph.items():
            for to_layer in to_layers:
                digraph.edge(
                    layer_uid(from_layer),
                    layer_uid(to_layer),
                    label=" {}".format(shapes_per_layer[from_layer]))

            if not to_layers:
                output = 'output-{}'.format(output_id)

                digraph.node(output, 'Output #{}'.format(output_id))
                digraph.edge(
                    layer_uid(from_layer), output,
                    label=" {}".format(shapes_per_layer[from_layer]))

                output_id += 1

        digraph.render(filepath, view=True)

    def get_params(self):
        return {'forward_graph': self.forward_graph}

    def __contains__(self, entity):
        return entity in self.forward_graph

    def __len__(self):
        return len(self.forward_graph)

    def __iter__(self):
        for layer in topological_sort(self.backward_graph):
            yield layer

    def __repr__(self):
        if not self.forward_graph:
            return "[empty graph]"

        def format_shapes(shape):
            if isinstance(shape, tf.TensorShape):
                return str(shape)

            shapes = ', '.join([format_shapes(s) for s in shape])
            return '[' + shapes + ']'

        return '{} -> [... {} layers ...] -> {}'.format(
            format_shapes(self.input_shape),
            len(self),
            format_shapes(self.output_shape))


def validate_graphs_before_combining(left_graph, right_graph):
    left_out_layers = left_graph.output_layers
    right_in_layers = right_graph.input_layers

    if len(left_out_layers) > 1 and len(right_in_layers) > 1:
        raise LayerConnectionError(
            "Cannot make many to many connection between graphs. One graph "
            "has {n_left_outputs} outputs (layer names: {left_names}) and "
            "the other one has {n_right_inputs} inputs (layer names: "
            "{right_names}). First graph: {left_graph}, Second graph: "
            "{right_graph}".format(
                left_graph=left_graph,
                n_left_outputs=len(left_out_layers),
                left_names=[layer.name for layer in left_out_layers],

                right_graph=right_graph,
                n_right_inputs=len(right_in_layers),
                right_names=[layer.name for layer in right_in_layers],
            )
        )

    left_out_shapes = as_tuple(left_graph.output_shape)
    right_in_shapes = as_tuple(right_graph.input_shape)

    for left_layer, left_out_shape in zip(left_out_layers, left_out_shapes):
        right = zip(right_in_layers, right_in_shapes)

        for right_layer, right_in_shape in right:
            if left_out_shape.is_compatible_with(right_in_shape):
                continue

            raise LayerConnectionError(
                "Cannot connect layer `{left_name}` to layer `{right_name}`, "
                "because output shape ({left_out_shape}) of the first layer "
                "is incompatible with the input shape ({right_in_shape}) "
                "of the second layer. First layer: {left_layer}, Second "
                "layer: {right_layer}".format(
                    left_layer=left_layer,
                    left_name=left_layer.name,
                    left_out_shape=left_out_shape,

                    right_layer=right_layer,
                    right_name=right_layer.name,
                    right_in_shape=right_in_shape,
                )
            )


def merge(left_graph, right_graph, combine=False):
    """
    Merges two graphs into single one. When ``combine=False`` new
    connection won't be created. And when ``combine=True`` input layers
    from the ``left_graph`` will be combined to the output layers from
    ``right_graph``.

    Parameters
    ----------
    left_graph : layer, network
    right_graph : layer, network

    combine : bool
        Defaults to ``False``.
    """
    if combine:
        validate_graphs_before_combining(left_graph, right_graph)

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
            "Cannot define connection between layers, because it creates "
            "cycle in the graph. Left graph: {}, Right graph: {}"
            "".format(left_graph, right_graph))

    return LayerGraph(forward_graph)


def parallel(*networks):
    """
    Merges all networks/layers into single network without joining
    input and output layers together.

    Parameters
    ----------
    *networks
        Layers or networks. Each network can be specified as a list or
        tuple. In this case, this input will be passed to the ``join``
        function in order to create full network from the specified
        list of the connections.

    Examples
    --------
    >>> from neupy.layers import *
    >>> network = parallel(
    ...     Input(10),
    ...     Input(9) >> Relu(5),
    ...     Relu(5),
    ... )
    [(?, 10), (?, 9), <unknown>] -> [... 4 layers ...] -> \
    [(?, 10), (?, 5), (?, 5)]

    Networks can be specified as a list of layers.

    >>> from neupy.layers import *
    >>> network = parallel([
    ...     Input((28, 28, 1)),
    ...     Convolution((3, 3, 16)),
    ... ], [
    ...     Input((28, 28, 1)),
    ...     Convolution((3, 3, 12)),
    ... ])
    >>>
    >>> network
    [(?, 28, 28, 1), (?, 28, 28, 1)] -> [... 4 layers ...] -> \
    [(?, 26, 26, 16), (?, 26, 26, 12)]
    """
    graph = LayerGraph()

    for network in networks:
        if isinstance(network, (list, tuple)):
            network = join(*network)
        graph = merge(graph, network)

    return graph


def join(*networks):
    """
    Sequentially combines layers and networks into single network.
    Function will go sequentially over layers/networks and
    combine output layers from the one network that it picks
    to the input layers from the next network in the sequence.

    Parameters
    ----------
    *networks
        Layers or networks.

    Examples
    --------
    >>> from neupy.layers import *
    >>> network = join(
    ...     Input((28, 28, 1)),
    ...     Convolution((3, 3, 16)) >> Relu(),
    ...     Convolution((3, 3, 16)) >> Relu(),
    ...     Reshape(),
    ...     Softmax(10),
    ... )
    >>> network
    (?, 28, 28, 1) -> [... 7 layers ...] -> (?, 10)
    """
    graph = LayerGraph()

    for network in networks:
        graph = merge(graph, network, combine=True)

    return graph


def repeat(network_or_layer, n):
    """
    Function copies input `n - 1` times and connects everything in sequential
    order.

    Parameters
    ----------
    network_or_layer : network or layer
        Layer or network (connection of layers).

    n : int
        Number of times input should be replicated.

    Examples
    --------
    >>> from neupy.layers import *
    >>>
    >>> block = Conv((3, 3, 32)) >> Relu() >> BN()
    >>> block
    <unknown> -> [... 3 layers ...] -> (?, ?, ?, 32)
    >>>
    >>> repeat(block, n=5)
    <unknown> -> [... 15 layers ...] -> (?, ?, ?, 32)
    """
    if n <= 0 or not isinstance(n, int):
        raise ValueError(
            "The `n` parameter should be a positive integer, "
            "got {} instead".format(n))

    if n == 1:
        return network_or_layer

    input_shape = network_or_layer.input_shape
    output_shape = network_or_layer.output_shape

    if not input_shape.is_compatible_with(output_shape):
        raise LayerConnectionError(
            "Cannot connect network/layer to its copy, because input "
            "shape is incompatible with the output shape. Input shape: {}, "
            "Output shape: {}".format(input_shape, output_shape))

    new_networks = [copy.deepcopy(network_or_layer) for _ in range(n - 1)]
    return join(network_or_layer, *new_networks)
