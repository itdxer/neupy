Parameter Initialization Methods
================================

This is a small topic that help you to understand how to initialize weights, bias and other parameters in NeuPy.

.. code-block:: python

    from neupy import layers, algorithms, init

    gdnet = algorithms.GradientDescent(
        [
            layers.Input(10),
            layers.Sigmoid(30, weight=init.HeNormal()),
            layers.Sigmoid(15, weight=init.HeNormal()),
        ]
    )

Basically we set up parameter equal to initialization method. So weight parameter will be defined based on the algorithm inside of the class.

Also it's possible to set up your own weight for layers.

.. code-block:: python

    import numpy as np
    from neupy import layers, algorithms

    gdnet = algorithms.GradientDescent(
        [
            layers.Input(10),
            layers.Sigmoid(30, weight=np.random.randn(10, 30)),
            layers.Sigmoid(15, weight=np.random.randn(30, 15)),
        ]
    )

More initialization methods you can find :ref:`here <init-methods>`.

Shared parameters between layers
--------------------------------

.. code-block:: python

    >>> from neupy import layers
    >>>
    >>> hidden_layer_1 = layers.Relu(10)
    >>> network = layers.Input(10) > hidden_layer_1
    >>>
    >>> hidden_layer_2 = layers.Relu(10, weight=hidden_layer_1.weight,
    ...                              bias=hidden_layer_1.bias)
    >>>
    >>> network = network > hidden_layer_2
    >>> network
    Input(10) > Relu(10) > Relu(10)

Create custom initialization methods
------------------------------------

It is possible to define custom initialization method.

.. code-block:: python

    import numpy as np
    from neupy import layers, algorithms, init

    class Exponential(init.Initializer):
        def __init__(self, scale=0.01):
            self.scale = scale

        def sample(self, shape):
            return np.random.exponential(self.scale, size=shape)

    gdnet = algorithms.GradientDescent(
        [
            layers.Input(10),
            layers.Sigmoid(30, weight=Exponential()),
            layers.Sigmoid(15, weight=Exponential()),
        ]
    )

Initialization class requires only the ``sample`` method that accepts ``shape`` argument and returns tensor with specified shape.
