import tempfile

import numpy as np

from neupy import storage, layers
from neupy.utils import asfloat

from base import BaseTestCase


class JSONStorageTestCase(BaseTestCase):
    def test_json_storage(self):
        connection_1 = layers.join(
            layers.Input(10),
            layers.parallel(
                layers.Sigmoid(5),
                layers.Relu(5),
            ),
            layers.Elementwise(),
        )
        connection_2 = layers.join(
            layers.Input(10),
            layers.parallel(
                layers.Sigmoid(5),
                layers.Relu(5),
            ),
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
