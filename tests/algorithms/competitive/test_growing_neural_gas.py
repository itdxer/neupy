import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from neupy import algorithms
from neupy.algorithms.competitive.growing_neural_gas import (
    NeuralGasGraph, NeuronNode)

from base import BaseTestCase


class GrowingNeuralGasTestCase(BaseTestCase):
    debug_plot = False

    def setUp(self):
        super(BaseTestCase, self).setUp()
        self.data, _ = make_blobs(
            n_samples=1000,
            n_features=2,
            centers=2,
            cluster_std=0.4,
            random_state=0
        )

    def make_plot(self, data, network):
        plt.figure(figsize=(5, 6))
        plt.scatter(*data.T, alpha=0.4)

        for node_1, node_2 in network.graph.edges:
            weights = np.concatenate([node_1.weight, node_2.weight])
            line, = plt.plot(*weights.T, color='black')
            plt.setp(line, color='black')

        plt.show()

    def test_neural_gas_exceptions(self):
        gng = algorithms.GrowingNeuralGas(n_inputs=2)

        with self.assertRaisesRegexp(ValueError, "more than 2 dimensions"):
            gng.train(np.random.random((10, 2, 1)), epochs=1)

        with self.assertRaisesRegexp(ValueError, "have 2 features, but got 1"):
            gng.train(np.random.random((10, 1)), epochs=1)

    def test_simple_neural_gas(self):
        gng = algorithms.GrowingNeuralGas(
            n_inputs=2,
            n_start_nodes=2,
            shuffle_data=True,

            step=0.2,
            neighbour_step=0.05,
            max_edge_age=10,
            n_iter_before_neuron_added=50,

            error_decay_rate=0.995,
            after_split_error_decay_rate=0.5,
            min_distance_for_update=0.1,
            max_nodes=100,
            verbose=False,
        )
        data = self.data
        gng.train(data, epochs=10)

        # Add one useless node to make sure that it
        # would be deleted at the end
        first_node = gng.graph.nodes[0]
        new_node = algorithms.NeuronNode(weight=np.array([[1000., 1000.]]))
        new_node.useless_node = True

        gng.graph.add_node(new_node)
        gng.graph.add_edge(new_node, first_node)

        # We run it twice in order to make sure that we won't
        # reset weights on the second run.
        gng.train(data, epochs=10)

        if self.debug_plot:
            self.make_plot(data, gng)

        self.assertEqual(len(gng.graph.nodes), gng.max_nodes)
        self.assertAlmostEqual(gng.training_errors[-1], 0.09, places=2)

        useless_node_present = any(
            getattr(node, 'useless_node', False) for node in gng.graph.nodes)

        self.assertFalse(useless_node_present)

        with self.assertRaises(NotImplementedError):
            gng.predict(data)

        # Check that we can stop training in case if we don't get any updates
        before_epochs = gng.last_epoch
        gng.min_distance_for_update = 10
        gng.train(data, epochs=10)
        self.assertEqual(before_epochs + 1, gng.last_epoch)

    def test_gng_storage(self):
        gng = algorithms.GrowingNeuralGas(
            n_inputs=2,
            step=0.2,
            verbose=False,
            n_iter_before_neuron_added=5 * len(self.data),
        )
        gng.train(self.data, epochs=10)
        self.assertEqual(gng.graph.n_nodes, 4)

        gng_recovered = pickle.loads(pickle.dumps(gng))
        self.assertEqual(gng_recovered.graph.n_nodes, 4)

        gng_recovered.train(self.data, epochs=10)
        self.assertEqual(gng_recovered.graph.n_nodes, 6)

    def test_gng_repr(self):
        gng = algorithms.GrowingNeuralGas(
            n_inputs=2,
            step=0.2,
            verbose=True,
        )
        gng.train(self.data, epochs=10)
        self.assertEqual(
            str(gng.graph),
            "<NeuralGasGraph n_nodes=12, n_edges=19>",
        )


class NeuralGasGraphTestCase(BaseTestCase):
    def test_simple_graph(self):
        graph = NeuralGasGraph()

        node_a = NeuronNode(1)
        node_b = NeuronNode(2)
        node_c = NeuronNode(3)

        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.add_node(node_c)

        graph.add_edge(node_a, node_b)
        graph.add_edge(node_b, node_c)

        self.assertEqual(graph.n_nodes, 3)
        self.assertEqual(graph.n_edges, 2)

        self.assertEqual(graph.find_edge_id(node_a, node_b), (node_a, node_b))
        self.assertEqual(graph.find_edge_id(node_b, node_a), (node_a, node_b))

        error_message = "Edge between specified nodes doesn't exist"
        with self.assertRaisesRegexp(ValueError, error_message):
            graph.find_edge_id(node_a, node_c)

        graph.remove_edge(node_c, node_b)

        self.assertEqual(graph.n_nodes, 3)
        self.assertEqual(graph.n_edges, 1)

        graph.remove_node(node_c)

        self.assertEqual(graph.n_nodes, 2)
        self.assertEqual(graph.n_edges, 1)

        error_message = (
            "Cannot remove node, because it's "
            "connected to 1 other node\(s\)"
        )
        with self.assertRaisesRegexp(ValueError, error_message):
            graph.remove_node(node_a)

    def test_node_repr(self):
        node_a = NeuronNode(1)
        self.assertEqual(str(node_a), "<NeuronNode error=0.0>")

        node_a.error = 1
        self.assertEqual(str(node_a), "<NeuronNode error=1.0>")

        node_a.error = 3.141592654
        self.assertEqual(str(node_a), "<NeuronNode error=3.141593>")
