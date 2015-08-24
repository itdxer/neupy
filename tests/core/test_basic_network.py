from neuralpy.layers import SigmoidLayer, OutputLayer
from neuralpy.algorithms import Backpropagation

from base import BaseTestCase


class BasicNetworkTestCase(BaseTestCase):
    def test_network_attrs(self):
        network = Backpropagation((2, 2, 1))
        network.step = 0.1
        network.bias = True
        network.error = lambda x: x
        network.shuffle_data = True

        with self.assertRaises(TypeError):
            network.step = '33'

        with self.assertRaises(TypeError):
            network.use_bias = 123

        with self.assertRaises(TypeError):
            network.error = 'not a function'

        with self.assertRaises(TypeError):
            network.shuffle_data = 1

    def test_list_of_layers(self):
        bpnet = Backpropagation([SigmoidLayer(2), SigmoidLayer(3),
                                 SigmoidLayer(1), OutputLayer(10)])
        self.assertEqual([layer.input_size for layer in bpnet.layers],
                         [2, 3, 1, 10])
