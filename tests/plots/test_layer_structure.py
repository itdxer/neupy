import os
import sys
import random
import tempfile
from contextlib import contextmanager

from neupy import plots, layers, algorithms

from base import BaseTestCase


@contextmanager
def reproducible_mktemp():
    name = tempfile.mktemp()
    real_mktemp = tempfile.mktemp
    tempfile.mktemp = lambda *args, **kwargs: name
    yield
    tempfile.mktemp = real_mktemp


class LayerSturcturePlotTestCase(BaseTestCase):
    def test_that_layer_structure_does_not_fail(self):
        connection = layers.Input(10) > layers.Sigmoid(1)

        with tempfile.NamedTemporaryFile() as temp:
            filesize_before = os.path.getsize(temp.name)
            plots.layer_structure(connection, filepath=temp.name, show=False)
            filesize_after = os.path.getsize(temp.name)

            self.assertEqual(filesize_before, 0)
            self.assertGreater(filesize_after, filesize_before)

    def test_that_layer_structure_for_network(self):
        connection = layers.Input(10) > layers.Sigmoid(1)
        network = algorithms.GradientDescent(connection)

        with tempfile.NamedTemporaryFile() as temp:
            filesize_before = os.path.getsize(temp.name)
            plots.layer_structure(network, filepath=temp.name, show=False)
            filesize_after = os.path.getsize(temp.name)

            self.assertEqual(filesize_before, 0)
            self.assertGreater(filesize_after, filesize_before)

    def test_layer_structure_for_one_layer(self):
        layer = layers.Input(10)

        with tempfile.NamedTemporaryFile() as temp:
            filesize_before = os.path.getsize(temp.name)
            plots.layer_structure(layer, filepath=temp.name, show=False)
            filesize_after = os.path.getsize(temp.name)

            self.assertEqual(filesize_before, 0)
            self.assertEqual(filesize_after, 0)

    def test_layer_structure_undefined_file_name(self):
        connection = layers.Input(10) > layers.Sigmoid(1)

        with reproducible_mktemp():
            plots.layer_structure(connection, filepath=None, show=False)

            temp_filename = tempfile.mktemp()
            filesize_after = os.path.getsize(temp_filename)
            self.assertGreater(filesize_after, 0)

    def test_layer_structure_graphviz_import_error(self):
        connection = layers.Input(10) > layers.Sigmoid(1)

        system_pathes = sys.path
        # Make sure that python cannot find graphviz library
        sys.path = []

        # Delete it from module cache in case of it exists
        if 'graphviz' in sys.modules:
            del sys.modules['graphviz']

        with self.assertRaises(ImportError):
            plots.layer_structure(connection, filepath=None, show=False)

        sys.path = system_pathes
