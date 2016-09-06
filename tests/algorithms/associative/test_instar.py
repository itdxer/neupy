import numpy as np

from neupy import algorithms

from base import BaseTestCase


input_data = np.array([
    [0, 1, -1, -1],
    [1, 1, -1, -1],
])


class HebbRuleTestCase(BaseTestCase):
    def setUp(self):
        super(HebbRuleTestCase, self).setUp()
        self.default_properties = dict(
            n_inputs=4,
            n_outputs=1,
            n_unconditioned=1,
            weight=np.array([[3, 0, 0, 0]]).T,
        )

    def test_learning_process(self):
        inet = algorithms.Instar(
            step=1,
            verbose=False,
            **self.default_properties
        )

        inet.train(input_data, epochs=10)

        test_input = np.array([[0, 1, -1, -1]])
        self.assertEqual(inet.predict(test_input), 1)

        np.testing.assert_array_equal(
            inet.weight,
            np.array([[3, 1, -1, -1]]).T
        )

    def test_multiple_outputs(self):
        input_data = np.array([
            [-0.1961, 0.9806],
        ])
        innet = algorithms.Instar(
            n_inputs=2,
            n_outputs=3,
            n_unconditioned=1,
            weight=np.array([
                [0.7071, 0.7071, -1],
                [-0.7071, 0.7071, 0],
            ]),
            step=0.5,
            verbose=False
        )
        innet.train(input_data, epochs=1)
        np.testing.assert_array_almost_equal(
            innet.weight,
            np.array([
                [0.7071, 0.7071, -1],
                [-0.5704, 0.8439, 0.1368]
            ]),
            decimal=4
        )

    def test_train_different_inputs(self):
        self.assertInvalidVectorTrain(
            algorithms.Instar(
                step=1,
                verbose=False,
                **self.default_properties
            ),
            np.array([[0, 1, -1, -1]]),
            is_feature1d=False,
        )

    def test_predict_different_inputs(self):
        inet = algorithms.Instar(
            step=1,
            verbose=False,
            **self.default_properties
        )

        inet.train(input_data, epochs=10)
        self.assertInvalidVectorPred(inet, np.array([0, 1, -1, -1]), 1,
                                     is_feature1d=False)
