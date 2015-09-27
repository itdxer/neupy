import numpy as np

from neupy.algorithms import HebbRule
from neupy.layers import StepLayer, OutputLayer, SigmoidLayer

from base import BaseTestCase


input_data = np.array([
    [0, 1],
    [1, 1],
])


class HebbRuleTestCase(BaseTestCase):
    def setUp(self):
        super(HebbRuleTestCase, self).setUp()
        self.conn = StepLayer(2) > OutputLayer(1)

    def test_validations(self):
        with self.assertRaises(ValueError):
            # Wrong: too many layers
            HebbRule(
                SigmoidLayer(2) > SigmoidLayer(2) > OutputLayer(2),
                n_unconditioned=1
            )

        with self.assertRaises(AttributeError):
            # Wrong: Algorithm is not converge
            hn = HebbRule(self.conn)
            hn.train(input_data, epsilon=1e-5)

        with self.assertRaises(ValueError):
            # Wrong: Only step layers in connections
            HebbRule(
                SigmoidLayer(2) > OutputLayer(2),
                n_unconditioned=2
            )

        with self.assertRaises(ValueError):
            # Wrong: #features must be bigger than #unconditioned features.
            HebbRule(
                StepLayer(2) > OutputLayer(2),
                n_unconditioned=2
            )

    def test_learning_process(self):
        hn = HebbRule(
            self.conn,
            n_unconditioned=1,
            use_bias=True,
            step=1,
        )

        hn.train(input_data, epochs=2)

        self.assertEqual(hn.predict(np.array([[0, 0]]))[0, 0], 0)
        self.assertEqual(hn.predict(np.array([[0, 1]]))[0, 0], 1)
        self.assertEqual(hn.predict(np.array([[1, 0]]))[0, 0], 1)
        self.assertEqual(hn.predict(np.array([[1, 1]]))[0, 0], 1)

    def test_with_weight_decay(self):
        hn = HebbRule(
            self.conn,
            n_unconditioned=1,
            step=1,
            decay_rate=0.1
        )

        # Test learning limit
        hn.train(input_data, epochs=50)
        self.assertEqual(np.round(hn.input_layer.weight[1, 0], 2), 10)

        hn.train(input_data, epochs=50)
        self.assertEqual(np.round(hn.input_layer.weight[1, 0], 2), 10)

    def test_weights(self):
        # Test default weights
        hn = HebbRule(
            StepLayer(5) > OutputLayer(1),
            n_unconditioned=2,
        )
        np.testing.assert_array_equal(
            hn.input_layer.weight,
            np.array([[1, 1, 0, 0, 0]]).T
        )

        # Test custom weights
        random_weight = np.random.random((5, 1))
        hn = HebbRule(
            StepLayer(5, weight=random_weight) > OutputLayer(1),
            n_unconditioned=2,
        )
        np.testing.assert_array_equal(hn.input_layer.weight, random_weight)
