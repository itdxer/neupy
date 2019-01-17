from neupy import algorithms

from base import BaseTestCase
from data import simple_classification


class StepDecayTestCase(BaseTestCase):
    def test_simple_step_decay(self):
        x_train, x_test, y_train, y_test = simple_classification()
        optimizer = algorithms.Momentum(
            (10, 5, 1),
            step=algorithms.step_decay(
                initial_value=0.1,
                reduction_freq=10,
            ),
            momentum=0.99,
            batch_size='full',
            verbose=False,
            nesterov=True,
        )

        step = self.eval(optimizer.variables.step)
        self.assertAlmostEqual(step, 0.1)

        optimizer.train(x_train, y_train, x_test, y_test, epochs=31)

        step = self.eval(optimizer.variables.step)
        self.assertAlmostEqual(step, 0.1 / 4)
