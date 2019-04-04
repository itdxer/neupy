Layer definitions
=================

It's common that different papers might have different configurations for some layers, but they will refer to it in the same way. For example, saying that network uses convolutional layers, doesn't tell us much about their configurations, since convolutional layer might have some paddings or initialization for weights might be different. In order to solve this problem, NeuPy allows to customize layer's definition.

.. code-block:: python

    from neupy import init
    from neupy.layers import *

    Conv = Convolution.define(
        padding='SAME',
        weight=init.XavierNormal(),
        bias=None,  # no bias
    )
    BN = BatchNorm.define(
        epsilon=1e-7,
        alpha=0.001,
    )

    network = join(
        Input((32, 32, 3)),

        Conv((3, 3, 16)) >> Relu() >> BN(),
        Conv((3, 3, 16)) >> Relu() >> BN(),
        MaxPooling((2, 2)),

        Conv((3, 3, 64)) >> Relu() >> BN(),
        Conv((3, 3, 64)) >> Relu() >> BN(),
        MaxPooling((2, 2)),

        Reshape(),
        Softmax(10),
    )
