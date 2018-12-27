from neupy import algorithms, layers

from base import BaseTestCase
from helpers import simple_classification


class StepUpdateTestCase(BaseTestCase):
    def assert_invalid_step_values(self, step, initial_value,
                                   final_value, epochs):

        x_train, x_test, y_train, y_test = simple_classification()
        optimizer = algorithms.Momentum(
            [
                layers.Input(10),
                layers.Sigmoid(5),
                layers.Sigmoid(1),
            ],
            step=step,
            momentum=0.99,
            batch_size=None,
            verbose=False,
            nesterov=True,
        )

        step = self.eval(optimizer.variables.step)
        self.assertAlmostEqual(step, initial_value)

        optimizer.train(x_train, y_train, x_test, y_test, epochs=epochs)

        step = self.eval(optimizer.variables.step)
        self.assertAlmostEqual(step, final_value)

    def test_step_decay(self):
        self.assert_invalid_step_values(
            algorithms.step_decay(
                initial_value=0.1,
                reduction_freq=10,
            ),
            initial_value=0.1,
            final_value=0.1 / 4,
            epochs=31,
        )

    def test_exponential_decay(self):
        self.assert_invalid_step_values(
            algorithms.exponential_decay(
                initial_value=0.1,
                reduction_freq=10,
                reduction_rate=0.9,
            ),
            initial_value=0.1,
            final_value=0.1 * 0.9 ** 3,
            epochs=31,
        )

    def test_polynomial_decay_limit_reached(self):
        self.assert_invalid_step_values(
            algorithms.polynomial_decay(
                initial_value=0.1,
                decay_iter=20,
                minstep=0.02,
            ),
            initial_value=0.1,
            final_value=0.02,
            epochs=31,
        )

    def test_polynomial_decay_limit_not_reached(self):
        self.assert_invalid_step_values(
            algorithms.polynomial_decay(
                initial_value=0.1,
                decay_iter=40,
                minstep=0.02,
            ),
            initial_value=0.1,
            final_value=0.04,
            epochs=31,
        )
