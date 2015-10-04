import numpy as np

from neupy import algorithms
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
            algorithms.HebbRule(
                SigmoidLayer(2) > SigmoidLayer(2) > OutputLayer(2),
                n_unconditioned=1,
                verbose=False
            )

        with self.assertRaises(AttributeError):
            # Wrong: Algorithm is not converge
            hn = algorithms.HebbRule(self.conn, verbose=False)
            hn.train(input_data, epsilon=1e-5)

        with self.assertRaises(ValueError):
            # Wrong: Only step layers in connections
            algorithms.HebbRule(
                SigmoidLayer(2) > OutputLayer(2),
                n_unconditioned=2,
                verbose=False
            )

        with self.assertRaises(ValueError):
            # Wrong: #features must be bigger than #unconditioned features.
            algorithms.HebbRule(
                StepLayer(2) > OutputLayer(2),
                n_unconditioned=2,
                verbose=False
            )

    def test_learning_process(self):
        hn = algorithms.HebbRule(
            self.conn,
            n_unconditioned=1,
            use_bias=True,
            step=1,
            verbose=False,
        )

        hn.train(input_data, epochs=2)

        test_data = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ])
        np.testing.assert_array_equal(
            hn.predict(test_data),
            np.array([[0, 1, 1, 1]]).T
        )

    def test_with_weight_decay(self):
        hn = algorithms.HebbRule(
            self.conn,
            n_unconditioned=1,
            step=1,
            verbose=False,
            decay_rate=0.1,
        )

        # Test learning limit
        hn.train(input_data, epochs=50)
        self.assertEqual(np.round(hn.input_layer.weight[1, 0], 2), 10)

        hn.train(input_data, epochs=50)
        self.assertEqual(np.round(hn.input_layer.weight[1, 0], 2), 10)

    def test_weights(self):
        # Test default weights
        hn = algorithms.HebbRule(
            StepLayer(5) > OutputLayer(1),
            n_unconditioned=2,
            verbose=False,
        )
        np.testing.assert_array_equal(
            hn.input_layer.weight,
            np.array([[1, 1, 0, 0, 0]]).T
        )

        # Test custom weights
        random_weight = np.random.random((5, 1))
        hn = algorithms.HebbRule(
            StepLayer(5, weight=random_weight) > OutputLayer(1),
            n_unconditioned=2,
            verbose=False,
        )
        np.testing.assert_array_equal(hn.input_layer.weight, random_weight)

    def test_train_different_inputs(self):
        self.assertInvalidVectorTrain(
            algorithms.HebbRule(
                self.conn,
                n_unconditioned=1,
                step=1,
                verbose=False
            ),
            np.array([[0, 1]]),
            row1d=True,
        )

    def test_predict_different_inputs(self):
        inet = algorithms.HebbRule(
            self.conn,
            n_unconditioned=1,
            step=1,
            verbose=False
        )

        inet.train(input_data, epochs=10)
        self.assertInvalidVectorPred(inet, np.array([0, 0]), 0,
                                     row1d=True)
