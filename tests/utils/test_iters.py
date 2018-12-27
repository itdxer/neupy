import numpy as np

from neupy.utils import iters
from neupy.utils.iters import (
    average_batch_errors,
    count_samples,
    count_minibatches,
)

from base import BaseTestCase


class ItersUtilsTestCase(BaseTestCase):
    def test_minibatches(self):
        n_samples = 50
        batch_size = 20
        expected_shapes = [(20, 2), (20, 2), (10, 2)]

        data = np.random.random((n_samples, 2))
        batch_slices = list(iters.minibatches(data, batch_size))

        for batch, expected_shape in zip(batch_slices, expected_shapes):
            self.assertEqual(batch.shape, expected_shape)

    def test_minibatches_unknown_batch_size(self):
        data = np.random.random((24, 2))
        iterbatches = iters.minibatches(data, batch_size=None, shuffle=False)

        for index, batch in enumerate(iterbatches, start=1):
            self.assertEqual(batch.shape, (24, 2))
            self.assertIs(batch, data)

        self.assertEqual(index, 1)

    def test_minibatches_with_shuffle(self):
        data = np.arange(24)
        iterbatches = iters.minibatches(data, batch_size=12, shuffle=True)

        collected_samples = []
        for batch in iterbatches:
            collected_samples.append(batch)

        collected_samples = np.concatenate(collected_samples)
        np.testing.assert_array_equal(data, sorted(collected_samples))
        self.assertFalse(np.allclose(data == collected_samples, b=1e-7))

    def test_minibatches_nested_inputs(self):
        data = [np.arange(24)], np.arange(24)
        iterbatches = iters.minibatches(data, batch_size=12, shuffle=False)

        collected_samples = []
        for batch in iterbatches:
            collected_samples.append(batch)

        batch_1 = np.arange(12)
        batch_2 = np.arange(12, 24)

        self.assertEqual(len(collected_samples), 2)
        np.testing.assert_array_equal(collected_samples[0][0], [batch_1])
        np.testing.assert_array_equal(collected_samples[0][1], batch_1)

        np.testing.assert_array_equal(collected_samples[1][0], [batch_2])
        np.testing.assert_array_equal(collected_samples[1][1], batch_2)

    def test_minibatches_nested_inputs_with_nones(self):
        data = [np.arange(24)], None
        iterbatches = iters.minibatches(data, batch_size=12, shuffle=False)

        collected_samples = []
        for batch in iterbatches:
            collected_samples.append(batch)

        batch_1 = np.arange(12)
        batch_2 = np.arange(12, 24)

        self.assertEqual(len(collected_samples), 2)
        np.testing.assert_array_equal(collected_samples[0][0], [batch_1])
        np.testing.assert_array_equal(collected_samples[0][1], None)

        np.testing.assert_array_equal(collected_samples[1][0], [batch_2])
        np.testing.assert_array_equal(collected_samples[1][1], None)

    def test_apply_batches_with_progressbar(self):
        # So far we just make sure that test didn't trigger any error
        # In the future, we need to check content of the terminal output
        outputs = iters.apply_batches(
            function=lambda x: x * 2,
            inputs=np.arange(20),
            batch_size=8,
            show_progressbar=True,
        )
        self.assertEqual(len(outputs), 3)

    def test_apply_batches(self):
        def mse(y_actual, y_predicted):
            return np.mean((y_actual - y_predicted) ** 2)

        y_actual = np.arange(20)
        y_predicted = np.ones(20) * 10

        outputs = iters.apply_batches(
            function=mse,
            inputs=[y_actual, y_predicted],
            batch_size=7,
        )
        np.testing.assert_array_almost_equal(
            outputs, np.array([53, 4, 45.1666666]))

        avg_loss = iters.apply_batches(
            function=mse,
            inputs=[y_actual, y_predicted],
            batch_size=7,
            average_outputs=True,
        )
        self.assertEqual(avg_loss, mse(y_actual, y_predicted))

    def test_batch_average(self):
        expected_error = 0.9  # or 225 / 250
        actual_error = average_batch_errors([1, 1, 0.5], 250, 100)
        self.assertAlmostEqual(expected_error, actual_error)

        expected_error = 0.8  # or 240 / 300
        actual_error = average_batch_errors([1, 1, 0.4], 300, 100)
        self.assertAlmostEqual(expected_error, actual_error)

    def test_count_samples_function(self):
        x = np.random.random((10, 5))

        self.assertEqual(count_samples(x), 10)
        self.assertEqual(count_samples([x, x]), 10)
        self.assertEqual(count_samples([[x], None]), 10)

    def test_count_minibatches(self):
        x = np.random.random((10, 5))

        self.assertEqual(count_minibatches(x, batch_size=2), 5)
        self.assertEqual(count_minibatches(x, batch_size=3), 4)
        self.assertEqual(count_minibatches([x], batch_size=5), 2)
        self.assertEqual(count_minibatches([[x], None], batch_size=4), 3)
