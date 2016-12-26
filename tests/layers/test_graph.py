import textwrap

import numpy as np
import theano.tensor as T

from neupy import layers
from neupy.utils import asfloat
from neupy.exceptions import LayerConnectionError
from neupy.layers.connections.graph import LayerGraph, is_cyclic
from neupy.layers.connections.base import topological_sort

from base import BaseTestCase


class LayerGraphTestCase(BaseTestCase):
    def test_tree_graph(self):
        l0 = layers.Input(1)
        l1 = layers.Sigmoid(10)
        l2 = layers.Sigmoid(20)
        l3 = layers.Sigmoid(30)
        l4 = layers.Sigmoid(40)
        l5 = layers.Sigmoid(50)
        l6 = layers.Sigmoid(60)

        # Tree Structure:
        #
        # l0 - l1 - l5 - l6
        #        \
        #         l2 - l4
        #           \
        #            -- l3
        graph = LayerGraph()
        # Connection #1
        graph.connect_layers(l0, l1)
        graph.connect_layers(l1, l5)
        graph.connect_layers(l5, l6)
        # Connection #2
        graph.connect_layers(l1, l2)
        graph.connect_layers(l2, l3)
        # Connection #3
        graph.connect_layers(l2, l4)

        for layer in graph.forward_graph:
            layer.initialize()

        subgraph = graph.subgraph_for_output(l6)
        self.assertEqual(1, len(subgraph.output_layers))
        self.assertIs(l6, subgraph.output_layers[0])
        self.assertEqual(1, len(subgraph.input_layers))
        self.assertIs(l0, subgraph.input_layers[0])

        x = T.matrix()
        outputs = graph.propagate_forward(x)

        text_input = asfloat(np.array([[1]]))
        expected_shapes = [(1, 30), (1, 40), (1, 60)]

        for output, expected_shape in zip(outputs, expected_shapes):
            output_value = output.eval({x: text_input})
            self.assertIn(output_value.shape, expected_shapes)

    def test_graph_cycles_error(self):
        l1 = layers.Input(10)
        l2 = layers.Sigmoid(20)
        l3 = layers.Sigmoid(30)

        graph = LayerGraph()
        graph.connect_layers(l1, l2)
        graph.connect_layers(l2, l3)

        with self.assertRaises(LayerConnectionError):
            graph.connect_layers(l3, l1)

    def test_is_cyclic_graph(self):
        graph1 = {1: [2], 2: [3], 3: [1]}
        self.assertTrue(is_cyclic(graph1), msg=graph1)

        graph2 = {1: [2], 2: [3], 3: [4]}
        self.assertFalse(is_cyclic(graph2), msg=graph2)

    def test_graph_connection_error(self):
        l1 = layers.Input(10)
        l2 = layers.Input(20)
        l3 = layers.Sigmoid(30)

        graph = LayerGraph()
        graph.connect_layers(l1, l3)

        with self.assertRaises(LayerConnectionError):
            graph.connect_layers(l2, l3)

        with self.assertRaises(LayerConnectionError):
            graph.connect_layers(l1, l1)

    def test_one_to_one_graph(self):
        l0 = layers.Input(1)
        l1 = layers.Sigmoid(10)
        l2 = layers.Sigmoid(20)
        l3 = layers.Sigmoid(30)
        l41 = layers.Sigmoid(40)
        l42 = layers.Sigmoid(40)
        le = layers.Elementwise()

        # Graph Structure:
        # l0 -> le
        #
        # l0 - l1 - l41 -- le
        #        \        /
        #         l2 - l42
        #           \
        #            -- l3
        graph = LayerGraph()

        # Connection #1
        graph.connect_layers(l0, l1)
        graph.connect_layers(l1, l41)
        graph.connect_layers(l41, le)

        graph.connect_layers(l1, l2)
        graph.connect_layers(l2, l42)
        graph.connect_layers(l42, le)

        # Connection #2
        graph.connect_layers(l2, l3)

        for layer in graph.forward_graph:
            layer.initialize()

        subgraph = graph.subgraph_for_output(le)
        self.assertIsNot(l3, subgraph.forward_graph)
        self.assertEqual(6, len(subgraph))

        # Input layers
        self.assertEqual(1, len(subgraph.input_layers))
        self.assertEqual([l0], subgraph.input_layers)

        # Output layers
        self.assertEqual(1, len(subgraph.output_layers))
        self.assertEqual([le], subgraph.output_layers)

        x = T.matrix()
        y = subgraph.propagate_forward(x)
        test_input = asfloat(np.array([[1]]))
        output = y.eval({x: test_input})

        self.assertEqual((1, 40), output.shape)

    def test_many_to_one_graph(self):
        l0 = layers.Input(1)
        l11 = layers.Sigmoid(10)
        le = layers.Elementwise()
        l3 = layers.Sigmoid(30)
        l4 = layers.Sigmoid(40)
        l5 = layers.Input(50)
        l6 = layers.Sigmoid(60)
        l12 = layers.Sigmoid(10)

        # Graph Structure:
        # [l0, l12] -> l6
        #
        # l0 - l11 - le - l6
        #           /
        #    l5 - l12 - l4
        #           \
        #            -- l3
        graph = LayerGraph()

        # Connection #1
        graph.connect_layers(l0, l11)
        graph.connect_layers(l11, le)
        graph.connect_layers(le, l6)

        graph.connect_layers(l5, l12)
        graph.connect_layers(l12, le)

        # Connection #2
        graph.connect_layers(l12, l4)

        # Connection #3
        graph.connect_layers(l12, l3)

        subgraph = graph.subgraph_for_output(l6)
        self.assertIsNot(l4, subgraph.forward_graph)
        self.assertIsNot(l3, subgraph.forward_graph)
        self.assertEqual(6, len(subgraph))

        # Input layers
        self.assertEqual(2, len(subgraph.input_layers))
        self.assertIn(l0, subgraph.input_layers)
        self.assertIn(l5, subgraph.input_layers)

        # Output layers
        self.assertEqual(1, len(subgraph.output_layers))
        self.assertEqual([l6], subgraph.output_layers)

    def test_many_to_many_graph(self):
        l0 = layers.Input(1)
        l11 = layers.Sigmoid(10)
        le = layers.Elementwise()
        l3 = layers.Sigmoid(30)
        l4 = layers.Sigmoid(40)
        l5 = layers.Sigmoid(50)
        l6 = layers.Sigmoid(60)
        l12 = layers.Sigmoid(70)

        # Graph Structure:
        # [l0, l12] -> [l6, l4]
        #
        # l0 - l11 - l5 - l6
        #        \
        #   l12 - le - l4
        #           \
        #            -- l3
        graph = LayerGraph()

        # Connection #1
        graph.connect_layers(l0, l11)
        graph.connect_layers(l11, l5)
        graph.connect_layers(l5, l6)

        # Connection #2
        graph.connect_layers(l11, le)
        graph.connect_layers(le, l4)

        # Connection #3
        graph.connect_layers(le, l3)

        # Connection #4
        graph.connect_layers(l12, le)

    def test_topological_sort_exception(self):
        cyclic_graph = {'a': ['b'], 'b': ['a']}
        with self.assertRaises(RuntimeError):
            topological_sort(cyclic_graph)

    def test_subgraph_by_layer(self):
        layer_1 = layers.Input(1)
        layer_2 = layers.Input(2)

        graph = LayerGraph()
        graph.add_layer(layer_1)

        subgraph = graph.subgraph_for_output(layer_1)
        self.assertEqual(len(subgraph.forward_graph), 1)

        subgraph = graph.subgraph_for_output(layer_2)
        self.assertEqual(len(subgraph.forward_graph), 0)

    def test_graph_propagate_forward(self):
        layer_1 = layers.Input(1)
        layer_2 = layers.Input(2)

        graph = LayerGraph()
        graph.add_layer(layer_1)

        with self.assertRaises(ValueError):
            graph.propagate_forward({layer_2: T.matrix()})

    def test_graph_connect_layer_missed_input_shapes(self):
        # input_layer_1 -> concatenate
        #                    /
        #              hidden_layer
        #                  /
        #         input_layer_2
        input_layer_1 = layers.Input(10)
        hidden_layer = layers.Sigmoid(10)
        input_layer_2 = layers.Input(2)
        merge_layer = layers.Concatenate()

        graph = LayerGraph()

        # First we join layers that doesn't have input shapes
        graph.connect_layers(hidden_layer, merge_layer)
        self.assertEqual(merge_layer.output_shape, None)

        # Now we can join layer that has input shape
        graph.connect_layers(input_layer_1, merge_layer)
        self.assertEqual(merge_layer.output_shape, None)

        # At this point we have fully constructed connection
        graph.connect_layers(input_layer_2, hidden_layer)
        self.assertEqual(merge_layer.output_shape, (20,))

    def test_graph_layer_connection_exception(self):
        input_layer_1 = layers.Input(10)
        input_layer_2 = layers.Input(20)

        graph = LayerGraph()

        with self.assertRaises(LayerConnectionError):
            graph.connect_layers(input_layer_1, input_layer_2)

    def test_graph_repr(self):
        l1 = layers.Input(1)
        l2 = layers.Sigmoid(2)
        l3 = layers.Sigmoid(3)
        l4 = layers.Sigmoid(4)

        graph = LayerGraph()

        graph.connect_layers(l1, l2)
        graph.connect_layers(l2, l3)
        graph.connect_layers(l3, l4)

        expected_output = textwrap.dedent("""
        [(Input(1), [Sigmoid(2)]),
         (Sigmoid(2), [Sigmoid(3)]),
         (Sigmoid(3), [Sigmoid(4)]),
         (Sigmoid(4), [])]
        """).strip()

        self.assertEqual(expected_output, repr(graph).strip())

    def test_graph_with_duplicated_layer_names_exception(self):
        graph = LayerGraph()

        layer_1 = layers.Sigmoid(1, name='sigmoid-1')
        layer_2 = layers.Sigmoid(1, name='sigmoid-1')

        with self.assertRaises(LayerConnectionError):
            graph.connect_layers(layer_1, layer_2)
