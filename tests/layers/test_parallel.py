import numpy as np
import theano
import theano.tensor as T

from neupy import layers
from neupy.utils import asfloat

from base import BaseTestCase


class ParallelLayerTestCase(BaseTestCase):
    def test_parallel_layer(self):
        input_layer = layers.Input((3, 8, 8))
        parallel_layer = layers.Parallel(
            [[
                layers.Convolution((11, 5, 5)),
            ], [
                layers.Convolution((10, 3, 3)),
                layers.Convolution((5, 3, 3)),
            ]],
            layers.Concatenate(),
        )

        conn = layers.join(input_layer, parallel_layer)

        x = T.tensor4()
        y = theano.function([x], conn.output(x))

        x_tensor4 = asfloat(np.random.random((10, 3, 8, 8)))
        output = y(x_tensor4)
        self.assertEqual(output.shape, (10, 11 + 5, 4, 4))
