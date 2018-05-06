import os
import tempfile

import numpy as np

from neupy import algorithms, layers, storage
from neupy.utils import asfloat
from neupy.exceptions import StopTraining

from base import BaseTestCase
from data import simple_classification


class LayerStoragePickleTestCase(BaseTestCase):
    def test_storage_pickle_save_conection_from_network(self):
        network = algorithms.GradientDescent([
            layers.Input(10),
            layers.Sigmoid(5),
            layers.Sigmoid(2),
        ])

        with tempfile.NamedTemporaryFile() as temp:
            storage.save_pickle(network, temp.name)
            temp.file.seek(0)

            filesize_after = os.path.getsize(temp.name)
            self.assertGreater(filesize_after, 0)

    def test_simple_storage_pickle(self):
        connection_1 = layers.join(
            layers.Input(10),
            layers.Sigmoid(5),
            layers.Sigmoid(2),
        )
        connection_2 = layers.join(
            layers.Input(10),
            layers.Sigmoid(5),
            layers.Sigmoid(2),
        )

        random_input = asfloat(np.random.random((13, 10)))
        random_output_1 = self.eval(connection_1.output(random_input))
        random_output_2_1 = self.eval(connection_2.output(random_input))

        self.assertFalse(np.any(random_output_1 == random_output_2_1))

        with tempfile.NamedTemporaryFile() as temp:
            storage.save_pickle(connection_1, temp.name)
            storage.load_pickle(connection_2, temp.name)
            random_output_2_2 = self.eval(connection_2.output(random_input))

            np.testing.assert_array_almost_equal(
                random_output_1, random_output_2_2)

    def test_storage_pickle_save_load_save(self):
        connection = layers.join(
            layers.Input(10),
            layers.Sigmoid(5),
            layers.Sigmoid(2),
        )

        with tempfile.NamedTemporaryFile() as temp:
            storage.save_pickle(connection, temp.name)
            temp.file.seek(0)

            filesize_first = os.path.getsize(temp.name)

            storage.load_pickle(connection, temp.name)

        with tempfile.NamedTemporaryFile() as temp:
            storage.save_pickle(connection, temp.name)
            temp.file.seek(0)

            filesize_second = os.path.getsize(temp.name)

        self.assertEqual(filesize_first, filesize_second)

    def test_storage_pickle_save_and_load_during_the_training(self):
        tempdir = tempfile.mkdtemp()
        x_train, x_test, y_train, y_test = simple_classification()

        errors = {}

        def on_epoch_end(network):
            epoch = network.last_epoch
            errors[epoch] = network.prediction_error(x_test, y_test)

            if epoch == 4:
                storage.load_pickle(
                    network.connection,
                    os.path.join(tempdir, 'training-epoch-2'))
                raise StopTraining('Stop training process after 4th epoch')
            else:
                storage.save_pickle(
                    network.connection,
                    os.path.join(tempdir, 'training-epoch-{}'.format(epoch)))

        gdnet = algorithms.GradientDescent(
            connection=(10, 4, 1),
            epoch_end_signal=on_epoch_end,
            step=0.5
        )
        gdnet.train(x_train, y_train)

        validation_error = gdnet.prediction_error(x_test, y_test)

        self.assertGreater(errors[2], errors[4])
        self.assertAlmostEqual(validation_error, errors[2])
        self.assertNotAlmostEqual(validation_error, errors[4])
