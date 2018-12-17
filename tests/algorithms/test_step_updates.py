from neupy import algorithms

from base import BaseTestCase
from data import simple_classification


class StepUpdateTestCase(BaseTestCase):
    def assert_invalid_step_values(self, step, initial_value,
                                   final_value, epochs):

        x_train, x_test, y_train, y_test = simple_classification()
        optimizer = algorithms.Momentum(
            (10, 5, 1),
            step=step,
            momentum=0.99,
            batch_size='full',
            verbose=False,
            nesterov=True,
        )

        step = self.eval(optimizer.variables.step)
        self.assertAlmostEqual(step, initial_value)

        optimizer.train(x_train, y_train, x_test, y_test, epochs=epochs)

        step = self.eval(optimizer.variables.step)
        self.assertAlmostEqual(step, final_value)

    def test_simple_step_decay(self):
        self.assert_invalid_step_values(
            algorithms.step_decay(
                initial_value=0.1,
                reduction_freq=10,
            ),
            initial_value=0.1,
            final_value=0.1 / 4,
            epochs=31,
        )
