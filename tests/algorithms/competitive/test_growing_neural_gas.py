import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from neupy import algorithms

from base import BaseTestCase


class GrowingNeuralGasTestCase(BaseTestCase):
    debug_plot = False

    def make_plot(self, data, network):
        plt.figure(figsize=(5, 6))
        plt.scatter(*data.T, alpha=0.4)

        for node_1, node_2 in network.graph.edge_ages:
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
        data, _ = make_blobs(
            n_samples=1000,
            n_features=2,
            centers=2,
            cluster_std=0.4,
        )

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
            max_nodes=100,
            verbose=True
        )
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
        self.assertAlmostEqual(gng.errors.last(), 0.09, places=2)

        useless_node_present = any(
            getattr(node, 'useless_node', False) for node in gng.graph.nodes)

        self.assertFalse(useless_node_present)

        with self.assertRaises(NotImplementedError):
            gng.predict(data)
