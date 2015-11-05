import numpy as np

from neupy import algorithms, layers

from base import BaseTestCase


input_data = np.array([
    [0, 1, -1, -1],
    [1, 1, -1, -1],
])


class HebbRuleTestCase(BaseTestCase):
    def setUp(self):
        super(HebbRuleTestCase, self).setUp()
        kwargs = {'weight': np.array([[3, 0, 0, 0]]).T}
        self.conn = layers.Step(4, **kwargs) > layers.Output(1)

    def test_learning_process(self):
        inet = algorithms.Instar(
            self.conn,
            n_unconditioned=1,
            step=1,
            verbose=False
        )

        inet.train(input_data, epochs=10)

        test_input = np.array([[0, 1, -1, -1]])
        self.assertEqual(inet.predict(test_input), 1)

        np.testing.assert_array_equal(
            inet.input_layer.weight,
            np.array([[3, 1, -1, -1]]).T
        )

    def test_multiple_outputs(self):
        input_data = np.array([
            [-0.1961, 0.9806],
        ])
        input_size, output_size = (2, 3)
        input_layer = layers.Step(
            input_size,
            weight=np.array([
                [0.7071, 0.7071, -1],
                [-0.7071, 0.7071, 0],
            ])
        )
        output_layer = layers.CompetitiveOutput(output_size)
        hamming_network = algorithms.Instar(
            input_layer > output_layer,
            step=0.5,
            n_unconditioned=1,
            verbose=False
        )
        hamming_network.train(input_data, epochs=1)
        np.testing.assert_array_almost_equal(
            hamming_network.input_layer.weight,
            np.array([
                [0.7071, 0.7071, -1],
                [-0.5704, 0.8439, 0.1368]
            ]),
            decimal=4
        )


    def test_train_different_inputs(self):
        self.assertInvalidVectorTrain(
            algorithms.Instar(
                layers.Step(4) > layers.Output(1),
                n_unconditioned=1,
                step=1,
                verbose=False
            ),
            np.array([[0, 1, -1, -1]]),
            row1d=True,
        )

    def test_predict_different_inputs(self):
        inet = algorithms.Instar(
            self.conn,
            n_unconditioned=1,
            step=1,
            verbose=False
        )

        inet.train(input_data, epochs=10)
        self.assertInvalidVectorPred(inet, np.array([0, 1, -1, -1]), 1,
                                     row1d=True)
