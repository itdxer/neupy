import os
import tempfile
from contextlib import contextmanager
from collections import OrderedDict

from neupy import plots, layers, algorithms
from neupy.plots.network_structure import exclude_layer_from_graph

from base import BaseTestCase


@contextmanager
def reproducible_mktemp():
    name = tempfile.mktemp()
    real_mktemp = tempfile.mktemp
    tempfile.mktemp = lambda *args, **kwargs: name
    yield
    tempfile.mktemp = real_mktemp


class LayerStructurePlotTestCase(BaseTestCase):
    def test_that_network_structure_does_not_fail(self):
        network = layers.Input(10) > layers.Sigmoid(1)

        with tempfile.NamedTemporaryFile() as temp:
            filesize_before = os.path.getsize(temp.name)
            plots.network_structure(network, filepath=temp.name, show=False)
            filesize_after = os.path.getsize(temp.name)

            self.assertEqual(filesize_before, 0)
            self.assertGreater(filesize_after, filesize_before)

    def test_that_network_structure_for_network(self):
        network = layers.Input(10) > layers.Sigmoid(1)
        network = algorithms.GradientDescent(network)

        with tempfile.NamedTemporaryFile() as temp:
            filesize_before = os.path.getsize(temp.name)
            plots.network_structure(network, filepath=temp.name, show=False)
            filesize_after = os.path.getsize(temp.name)

            self.assertEqual(filesize_before, 0)
            self.assertGreater(filesize_after, filesize_before)

    def test_network_structure_undefined_file_name(self):
        network = layers.Input(10) > layers.Sigmoid(1)

        with reproducible_mktemp():
            plots.network_structure(network, filepath=None, show=False)

            temp_filename = tempfile.mktemp()
            filesize_after = os.path.getsize(temp_filename)
            self.assertGreater(filesize_after, 0)


class LayerStructureExcludeLayersPlotTestCase(BaseTestCase):
    def test_network_structure_exclude_layer_nothing_to_exclude(self):
        network = layers.Input(10) > layers.Sigmoid(1)
        graph = network.graph.forward_graph
        new_graph = exclude_layer_from_graph(graph, tuple())

        self.assertEqual(graph, new_graph)

    def test_network_structure_exclude_layer(self):
        input_layer = layers.Input(10)
        network = input_layer > layers.Sigmoid(1)
        graph = network.graph.forward_graph

        actual_graph = exclude_layer_from_graph(graph, [layers.Sigmoid])
        expected_graph = OrderedDict()
        expected_graph[input_layer] = []

        self.assertEqual(expected_graph, actual_graph)

    def test_network_structure_ignore_layers_attr(self):
        input_layer = layers.Input(10)
        network = input_layer > layers.Sigmoid(1)

        with tempfile.NamedTemporaryFile() as temp:
            plots.network_structure(network, filepath=temp.name, show=False,
                                    ignore_layers=[])
            filesize_first = os.path.getsize(temp.name)

        with tempfile.NamedTemporaryFile() as temp:
            plots.network_structure(network, filepath=temp.name, show=False,
                                    ignore_layers=[layers.Sigmoid])
            filesize_second = os.path.getsize(temp.name)

        # First one should have more layers to draw
        # than the second one
        self.assertGreater(filesize_first, filesize_second)
