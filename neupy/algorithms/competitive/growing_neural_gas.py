from operator import attrgetter

import numpy as np

from neupy.utils import format_data
from neupy.exceptions import StopTraining
from neupy.algorithms.base import BaseNetwork
from neupy.core.properties import (NumberProperty, ProperFractionProperty,
                                   IntProperty)


__all__ = ('GrowingNeuralGas', 'NeuralGasGraph', 'NeuronNode')


def sample_data_point(data, n=1):
    indices = np.random.choice(len(data), n)
    return data[indices, :]


class NeuralGasGraph(object):
    """
    Undirected graph structure that stores neural gas network's
    neurons and connections between them.

    Attributes
    ----------
    edges_per_node : dict
        Dictionary that where key is a unique node and value is a list
        of nodes that connection to this edge.

    edges : dict
        Dictonary that stores age per each connection. Ech key will have
        the following format: ``(node_1, node_2)``.

    nodes : list
        List of all nodes in the graph (read-only attribute).

    n_nodes : int
        Number of nodes in the network (read-only attribute).

    n_edges : int
        Number of edges in the network (read-only attribute).
    """
    def __init__(self):
        self.edges_per_node = {}
        self.edges = {}

    @property
    def nodes(self):
        return list(self.edges_per_node.keys())

    @property
    def n_nodes(self):
        return len(self.edges_per_node)

    @property
    def n_edges(self):
        return len(self.edges)

    def add_node(self, node):
        self.edges_per_node[node] = set()

    def remove_node(self, node):
        if self.edges_per_node[node]:
            raise ValueError(
                "Cannot remove node, because it's connected to "
                "{} other node(s)".format(len(self.edges_per_node[node])))

        del self.edges_per_node[node]

    def add_edge(self, node_1, node_2):
        if node_2 in self.edges_per_node[node_1]:
            return self.reset_edge(node_1, node_2)

        self.edges_per_node[node_1].add(node_2)
        self.edges_per_node[node_2].add(node_1)
        self.edges[(node_1, node_2)] = 0

    def reset_edge(self, node_1, node_2):
        edge_id = self.find_edge_id(node_1, node_2)
        self.edges[edge_id] = 0

    def remove_edge(self, node_1, node_2):
        edge_id = self.find_edge_id(node_1, node_2)

        self.edges_per_node[node_1].remove(node_2)
        self.edges_per_node[node_2].remove(node_1)

        del self.edges[edge_id]

    def find_edge_id(self, node_1, node_2):
        if (node_1, node_2) in self.edges:
            return (node_1, node_2)

        if (node_2, node_1) in self.edges:
            return (node_2, node_1)

        raise ValueError("Edge between specified nodes doesn't exist")

    def __repr__(self):
        return "<{} n_nodes={}, n_edges={}>".format(
            self.__class__.__name__,
            self.n_nodes, self.n_edges)


class NeuronNode(object):
    """
    Structure representes neuron in the Neural Gas algorithm.

    Attributes
    ----------
    weight : 2d-array
        Neuron's position in the space.

    error : float
        Error accumulated during the training.
    """
    def __init__(self, weight):
        self.weight = weight
        self.error = 0

    def __repr__(self):
        return "<{} error={}>".format(
            self.__class__.__name__,
            round(float(self.error), 6))


class GrowingNeuralGas(BaseNetwork):
    """
    Growing Neural Gas (GNG) algorithm.

    Current algorithm has two modifications that hasn't been mentioned
    in the paper, but they help to speed up training.

    - The ``n_start_nodes`` parameter provides possibility to increase
      number of nodes during initialization step. It's useful when
      algorithm takes a lot of time building up large amount of neurons.

    - The ``min_distance_for_update`` parameter allows to speed up
      training when some data samples has neurons very close to them. The
      ``min_distance_for_update`` parameter controls threshold for the
      minimum distance for which we will want to update weights.

    Parameters
    ----------
    n_inputs : int
        Number of features in each sample.

    n_start_nodes : int
        Number of nodes that algorithm generates from the data during
        the initialization step. Defaults to ``2``.

    step : float
        Step (learning rate) for the neuron winner. Defaults to ``0.2``.

    neighbour_step : float
        Step (learning rate) for the neurons that connected via edges
        with neuron winner. This value typically has to be smaller than
        ``step`` value. Defaults to ``0.05``.

    max_edge_age : int
        It means that if edge won't be updated for ``max_edge_age`` iterations
        than it would be removed. The larger the value the more updates we
        allow to do before removing edge. Defaults to ``100``.

    n_iter_before_neuron_added : int
        Each ``n_iter_before_neuron_added`` weight update algorithm add new
        neuron. The smaller the value the more frequently algorithm adds
        new neurons to the network. Defaults to ``1000``.

    error_decay_rate : float
        This error decay rate would be applied to every neuron in the
        graph after each training iteration. It ensures that old errors
        will be reduced over time. Defaults to ``0.995``.

    after_split_error_decay_rate : float
        This decay rate reduces error for neurons with largest errors
        after algorithm added new neuron. This value typically lower than
        ``error_decay_rate``. Defaults to ``0.5``.

    max_nodes : int
        Maximum number of nodes that would be generated during the training.
        This parameter won't stop training when maximum number of nodes
        will be exceeded. Defaults to ``1000``.

    min_distance_for_update : float
        Parameter controls for which neurons we want to apply updates.
        In case if euclidean distance between data sample and closest
        neurons will be less than the ``min_distance_for_update`` value than
        update would be skipped for this data sample. Setting value to zero
        will disable effect provided by this parameter. Defaults to ``0``.

    {BaseNetwork.show_epoch}

    {BaseNetwork.shuffle_data}

    {BaseNetwork.signals}

    {Verbose.verbose}

    Methods
    -------
    train(X_train, epochs=100)
        Network learns topological structure of the data. Learned
        structure will be stored in the ``graph`` attribute.

    {BaseSkeleton.fit}

    initialize_nodes(data)
        Network initializes nodes randomly sampling ``n_start_nodes``
        from the data. It would be applied automatically before
        the training in case if graph is empty.

        Note: Node re-initialization can reset network.

    Notes
    -----
    - Unlike other algorithms this network doesn't make predictions.
      Instead, it learns topological structure of the data in form of
      the graph. After that training, structure of the network can be
      extracted from the ``graph`` attribute.

    - In order to speed up training, it might be useful to increase
      the ``n_start_nodes`` parameter.

    - During the training it happens that nodes learn topological
      structure of one part of the data better than the other, mostly
      because of the different data sample density in different places.
      Increasing the ``min_distance_for_update`` can speed up training
      ignoring updates for the neurons that very close to the data sample.
      (below specified ``min_distance_for_update`` value). Training can be
      stopped in case if none of the neurons has been updated during
      the training epoch.

    Attributes
    ----------
    graph : NeuralGasGraph instance
        This attribute stores all neurons and connections between them
        in the form of undirected graph.

    {BaseNetwork.Attributes}

    Examples
    --------
    >>> from neupy import algorithms
    >>> from sklearn.datasets import make_blobs
    >>>
    >>> data, _ = make_blobs(
    ...     n_samples=1000,
    ...     n_features=2,
    ...     centers=2,
    ...     cluster_std=0.4,
    ... )
    >>>
    >>> neural_gas = algorithms.GrowingNeuralGas(
    ...     n_inputs=2,
    ...     shuffle_data=True,
    ...     verbose=True,
    ...     max_edge_age=10,
    ...     n_iter_before_neuron_added=50,
    ...     max_nodes=100,
    ... )
    >>> neural_gas.graph.n_nodes
    100
    >>> len(neural_gas.graph.edges)
    175
    >>> edges = list(neural_gas.graph.edges.keys())
    >>> neuron_1, neuron_2 = edges[0]
    >>>
    >>> neuron_1.weight
    array([[-6.77166299,  2.4121606 ]])
    >>> neuron_2.weight
    array([[-6.829309  ,  2.27839633]])

    References
    ----------
    [1] A Growing Neural Gas Network Learns Topologies, Bernd Fritzke
    """
    n_inputs = IntProperty(minval=1, required=True)
    n_start_nodes = IntProperty(minval=2, default=2)

    step = NumberProperty(default=0.2, minval=0)
    neighbour_step = NumberProperty(default=0.05, minval=0)
    max_edge_age = IntProperty(default=100, minval=1)
    max_nodes = IntProperty(default=1000, minval=1)

    n_iter_before_neuron_added = IntProperty(default=1000, minval=1)
    after_split_error_decay_rate = ProperFractionProperty(default=0.5)
    error_decay_rate = ProperFractionProperty(default=0.995)
    min_distance_for_update = NumberProperty(default=0.0, minval=0)

    def __init__(self, *args, **kwargs):
        super(GrowingNeuralGas, self).__init__(*args, **kwargs)
        self.n_updates = 0
        self.graph = NeuralGasGraph()

    def format_input_data(self, X):
        is_feature1d = self.n_inputs == 1
        X = format_data(X, is_feature1d)

        if X.ndim != 2:
            raise ValueError("Cannot make prediction, because input "
                             "data has more than 2 dimensions")

        n_samples, n_features = X.shape

        if n_features != self.n_inputs:
            raise ValueError("Input data expected to have {} features, "
                             "but got {}".format(self.n_inputs, n_features))

        return X

    def initialize_nodes(self, data):
        self.graph = NeuralGasGraph()

        for sample in sample_data_point(data, n=self.n_start_nodes):
            self.graph.add_node(NeuronNode(sample.reshape(1, -1)))

    def train(self, X_train, epochs=100):
        X_train = self.format_input_data(X_train)

        if not self.graph.nodes:
            self.initialize_nodes(X_train)

        return super(GrowingNeuralGas, self).train(
            X_train=X_train, y_train=None,
            X_test=None, y_test=None,
            epochs=epochs)

    def one_training_update(self, X_train, y_train=None):
        graph = self.graph
        step = self.step
        neighbour_step = self.neighbour_step

        max_nodes = self.max_nodes
        max_edge_age = self.max_edge_age

        error_decay_rate = self.error_decay_rate
        after_split_error_decay_rate = self.after_split_error_decay_rate
        n_iter_before_neuron_added = self.n_iter_before_neuron_added

        # We square this value, because we deal with
        # squared distances during the training.
        min_distance_for_update = np.square(self.min_distance_for_update)

        n_samples = len(X_train)
        total_error = 0
        did_update = False

        for sample in X_train:
            nodes = graph.nodes
            weights = np.concatenate([node.weight for node in nodes])

            distance = np.linalg.norm(weights - sample, axis=1)
            neuron_ids = np.argsort(distance)

            closest_neuron_id, second_closest_id = neuron_ids[:2]
            closest_neuron = nodes[closest_neuron_id]
            second_closest = nodes[second_closest_id]
            total_error += distance[closest_neuron_id]

            if distance[closest_neuron_id] < min_distance_for_update:
                continue

            self.n_updates += 1
            did_update = True

            closest_neuron.error += distance[closest_neuron_id]
            closest_neuron.weight += step * (sample - closest_neuron.weight)

            graph.add_edge(closest_neuron, second_closest)

            for to_neuron in list(graph.edges_per_node[closest_neuron]):
                edge_id = graph.find_edge_id(to_neuron, closest_neuron)
                age = graph.edges[edge_id]

                if age >= max_edge_age:
                    graph.remove_edge(to_neuron, closest_neuron)

                    if not graph.edges_per_node[to_neuron]:
                        graph.remove_node(to_neuron)

                else:
                    graph.edges[edge_id] += 1
                    to_neuron.weight += neighbour_step * (
                        sample - to_neuron.weight)

            time_to_add_new_neuron = (
                self.n_updates % n_iter_before_neuron_added == 0 and
                graph.n_nodes < max_nodes)

            if time_to_add_new_neuron:
                nodes = graph.nodes
                largest_error_neuron = max(nodes, key=attrgetter('error'))
                neighbour_neuron = max(
                    graph.edges_per_node[largest_error_neuron],
                    key=attrgetter('error'))

                largest_error_neuron.error *= after_split_error_decay_rate
                neighbour_neuron.error *= after_split_error_decay_rate

                new_weight = 0.5 * (
                    largest_error_neuron.weight + neighbour_neuron.weight
                )
                new_neuron = NeuronNode(weight=new_weight.reshape(1, -1))

                graph.remove_edge(neighbour_neuron, largest_error_neuron)
                graph.add_node(new_neuron)
                graph.add_edge(largest_error_neuron, new_neuron)
                graph.add_edge(neighbour_neuron, new_neuron)

            for node in graph.nodes:
                node.error *= error_decay_rate

        if not did_update and min_distance_for_update != 0 and n_samples > 1:
            raise StopTraining(
                "Distance between every data sample and neurons, closest "
                "to them, is less then {}".format(min_distance_for_update))

        return total_error / n_samples

    def predict(self, *args, **kwargs):
        raise NotImplementedError(
            "Growing Neural Gas algorithm doesn't make prediction. "
            "It only learns graph structure from the data "
            "(class has `graph` attribute). ")
