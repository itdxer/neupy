import numpy as np

from neupy.utils import asfloat
from neupy import architectures

from base import BaseTestCase


class Resnet50TestCase(BaseTestCase):
    def test_resnet50_architecture(self):
        resnet50 = architectures.resnet50()
        self.assertShapesEqual(resnet50.input_shape, (None, 224, 224, 3))
        self.assertShapesEqual(resnet50.output_shape, (None, 1000))

        random_input = asfloat(np.random.random((7, 224, 224, 3)))
        prediction = self.eval(resnet50.output(random_input))
        self.assertEqual(prediction.shape, (7, 1000))

    def test_resnet50_exceptions(self):
        with self.assertRaises(ValueError):
            architectures.resnet50(in_out_ratio=2)

    def test_resnet50_no_global_pooling(self):
        resnet50 = architectures.resnet50(include_global_pool=False)

        self.assertShapesEqual(resnet50.input_shape, (None, 224, 224, 3))
        self.assertShapesEqual(resnet50.output_shape, (None, 7, 7, 2048))

    def test_resnet50_spatial(self):
        resnet50 = architectures.resnet50(
            include_global_pool=False,
            in_out_ratio=8,
        )
        self.assertShapesEqual(resnet50.input_shape, (None, 224, 224, 3))
        self.assertShapesEqual(resnet50.output_shape, (None, 28, 28, 2048))

        random_input = asfloat(np.random.random((7, 224, 224, 3)))
        prediction = self.eval(resnet50.output(random_input))
        self.assertEqual(prediction.shape, (7, 28, 28, 2048))

    def test_resnet50_dilation_rates(self):
        resnet50 = architectures.resnet50(
            include_global_pool=False,
            in_out_ratio=8,
        )

        layer = resnet50.layer('res4d_branch2b')
        self.assertEqual(layer.dilation, (2, 2))

        layer = resnet50.layer('res5a_branch2b')
        self.assertEqual(layer.dilation, (2, 2))

        layer = resnet50.layer('res5b_branch2b')
        self.assertEqual(layer.dilation, (4, 4))
