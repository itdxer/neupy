import numpy as np

from neupy import algorithms

from base import BaseTestCase


input_data = np.array([
    [0, 1],
    [1, 1],
])


class HebbRuleTestCase(BaseTestCase):
    def test_validations(self):
        invalid_cases = (
            # Missed required parameters
            dict(),
            dict(n_inputs=2),
            dict(n_inputs=2, n_outputs=1),

            # n_inputs should be greater than n_unconditioned
            dict(n_inputs=2, n_outputs=2, n_unconditioned=2, verbose=False),

            # Invalid shapes for the arrays
            dict(n_inputs=2, n_outputs=1, n_unconditioned=1,
                 weight=np.array([1])),
            dict(n_inputs=2, n_outputs=1, n_unconditioned=1,
                 bias=np.array([1, 2, 3])),
        )

        for invalid_case_params in invalid_cases:
            with self.assertRaises(ValueError):
                algorithms.HebbRule(**invalid_case_params)

    def test_learning_process(self):
        hn = algorithms.HebbRule(
            n_inputs=2,
            n_outputs=1,
            n_unconditioned=1,
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
            n_inputs=2,
            n_outputs=1,
            n_unconditioned=1,
            step=1,
            verbose=False,
            decay_rate=0.1,
        )

        # Test learning limit
        hn.train(input_data, epochs=50)
        self.assertEqual(np.round(hn.weight[1, 0], 2), 10)

        hn.train(input_data, epochs=50)
        self.assertEqual(np.round(hn.weight[1, 0], 2), 10)

    def test_weights(self):
        # Test default weights
        hn = algorithms.HebbRule(
            n_inputs=5,
            n_outputs=1,
            n_unconditioned=2,
            verbose=False,
        )
        np.testing.assert_array_equal(
            hn.weight,
            np.array([[1, 1, 0, 0, 0]]).T
        )
        np.testing.assert_array_equal(
            hn.bias,
            np.array([-0.5])
        )

        # Test custom weights
        random_weight = np.random.random((5, 1))
        hn = algorithms.HebbRule(
            n_inputs=5,
            n_outputs=1,
            n_unconditioned=2,
            weight=random_weight,
            verbose=False,
        )
        np.testing.assert_array_equal(hn.weight, random_weight)

    def test_train_different_inputs(self):
        self.assertInvalidVectorTrain(
            algorithms.HebbRule(
                n_inputs=2,
                n_outputs=1,
                n_unconditioned=1,
                step=1,
                verbose=False
            ),
            np.array([[0, 1]]),
            is_feature1d=False,
        )

    def test_predict_different_inputs(self):
        inet = algorithms.HebbRule(
            n_inputs=2,
            n_outputs=1,
            n_unconditioned=1,
            step=1,
            verbose=False
        )

        inet.train(input_data, epochs=10)
        self.assertInvalidVectorPred(inet, np.array([0, 0]), 0,
                                     is_feature1d=False)
