import numpy as np

from neupy import algorithms
from neupy.layers import Step, Output, Sigmoid

from base import BaseTestCase


input_data = np.array([
    [0, 1],
    [1, 1],
])


class HebbRuleTestCase(BaseTestCase):
    def test_validations(self):
        with self.assertRaises(ValueError):
            # Wrong: too many layers
            algorithms.HebbRule(
                Sigmoid(2) > Sigmoid(2) > Output(2),
                n_unconditioned=1,
                verbose=False
            )

        with self.assertRaises(AttributeError):
            # Wrong: Algorithm is not converge
            hn = algorithms.HebbRule(Step(2) > Output(1), verbose=False)
            hn.train(input_data, epsilon=1e-5)

        with self.assertRaises(ValueError):
            # Wrong: Only step layers in connections
            algorithms.HebbRule(
                Sigmoid(2) > Output(2),
                n_unconditioned=2,
                verbose=False
            )

        with self.assertRaises(ValueError):
            # Wrong: #features must be bigger than #unconditioned features.
            algorithms.HebbRule(
                Step(2) > Output(2),
                n_unconditioned=2,
                verbose=False
            )

    def test_learning_process(self):
        hn = algorithms.HebbRule(
            Step(2) > Output(1),
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
            Step(2) > Output(1),
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
            Step(5) > Output(1),
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
            Step(5, weight=random_weight) > Output(1),
            n_unconditioned=2,
            verbose=False,
        )
        np.testing.assert_array_equal(hn.input_layer.weight, random_weight)

    def test_train_different_inputs(self):
        self.assertInvalidVectorTrain(
            algorithms.HebbRule(
                Step(2) > Output(1),
                n_unconditioned=1,
                step=1,
                verbose=False
            ),
            np.array([[0, 1]]),
            is_feature1d=False,
        )

    def test_predict_different_inputs(self):
        inet = algorithms.HebbRule(
            Step(2) > Output(1),
            n_unconditioned=1,
            step=1,
            verbose=False
        )

        inet.train(input_data, epochs=10)
        self.assertInvalidVectorPred(inet, np.array([0, 0]), 0,
                                     is_feature1d=False)
