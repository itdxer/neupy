import json
import tempfile

import numpy as np
from mock import patch

from neupy import storage, layers
from neupy.utils import asfloat

from base import BaseTestCase


class HDF5StorageTestCase(BaseTestCase):
    def test_simple_storage_hdf5(self):
        network_1 = layers.join(
            layers.Input(10),
            [
                layers.Sigmoid(5),
                layers.Relu(5),
            ],
            layers.Elementwise(),
        )
        network_2 = layers.join(
            layers.Input(10),
            [
                layers.Sigmoid(5),
                layers.Relu(5),
            ],
            layers.Elementwise(),
        )

        random_input = asfloat(np.random.random((13, 10)))
        random_output_1 = self.eval(network_1.output(random_input))
        random_output_2_1 = self.eval(network_2.output(random_input))

        # Outputs has to be different
        self.assertFalse(np.any(random_output_1 == random_output_2_1))

        with tempfile.NamedTemporaryFile() as temp:
            storage.save_hdf5(network_1, temp.name)
            storage.load_hdf5(network_2, temp.name)

            random_output_2_2 = self.eval(
                network_2.output(random_input))

            np.testing.assert_array_almost_equal(
                random_output_1, random_output_2_2)

    def test_hdf5_storage_broken_attributes(self):
        network = layers.Input(1) > layers.Relu(2, name='relu')
        json_loads = json.loads

        def break_json(value):
            if value == '"relu"':
                return json_loads("{")
            return json_loads(value)

        with tempfile.NamedTemporaryFile() as temp:
            storage.save_hdf5(network, temp.name)

            with patch('json.loads', side_effect=break_json):
                storage.load_hdf5(network, temp.name)
