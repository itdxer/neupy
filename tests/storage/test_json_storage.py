import tempfile
from textwrap import dedent

import numpy as np
from mock import patch

from neupy import storage, layers
from neupy.utils import asfloat
from neupy.storage import dump_with_fastest_json_module

from base import BaseTestCase


class JSONStorageTestCase(BaseTestCase):
    def test_ujson_custom_indentation_during_dump(self):
        # Changing the mode, because in Python 3 in would
        # be opened as binary file
        with tempfile.NamedTemporaryFile(mode='w+') as temp:
            dump_with_fastest_json_module(
                {'test': 'json'}, temp.file, indent=2)

            temp.file.seek(0)

            actual_output = temp.read()
            expected_output = dedent("""
            {
              "test":"json"
            }
            """).strip()  # remove final '\n' symbol

            self.assertEqual(actual_output, expected_output)

    def test_json_storage(self):
        connection_1 = layers.join(
            layers.Input(10),
            [
                layers.Sigmoid(5),
                layers.Relu(5),
            ],
            layers.Elementwise(),
        )
        connection_2 = layers.join(
            layers.Input(10),
            [
                layers.Sigmoid(5),
                layers.Relu(5),
            ],
            layers.Elementwise(),
        )

        random_input = asfloat(np.random.random((13, 10)))
        random_output_1 = self.eval(connection_1.output(random_input))
        random_output_2_1 = self.eval(connection_2.output(random_input))

        # Outputs have to be different
        self.assertFalse(np.any(random_output_1 == random_output_2_1))

        with tempfile.NamedTemporaryFile() as temp:
            storage.save_json(connection_1, temp.name)
            storage.load_json(connection_2, temp.name)

            random_output_2_2 = self.eval(
                connection_2.output(random_input))

            np.testing.assert_array_almost_equal(
                random_output_1, random_output_2_2)

    def test_json_storage_without_ujson(self):
        with patch('pkgutil.find_loader') as mock_find_loader:
            mock_find_loader.return_value = None

            connection_1 = layers.join(
                layers.Input(10),
                [
                    layers.Sigmoid(5),
                    layers.Relu(5),
                ],
                layers.Elementwise(),
            )
            connection_2 = layers.join(
                layers.Input(10),
                [
                    layers.Sigmoid(5),
                    layers.Relu(5),
                ],
                layers.Elementwise(),
            )

            random_input = asfloat(np.random.random((13, 10)))
            random_output_1 = self.eval(connection_1.output(random_input))
            random_output_2_1 = self.eval(connection_2.output(random_input))

            # Outputs has to be different
            self.assertFalse(np.any(random_output_1 == random_output_2_1))

            with tempfile.NamedTemporaryFile() as temp:
                storage.save_json(connection_1, temp.name)
                storage.load_json(connection_2, temp.name)

                random_output_2_2 = self.eval(
                    connection_2.output(random_input))

                np.testing.assert_array_almost_equal(
                    random_output_1, random_output_2_2)
