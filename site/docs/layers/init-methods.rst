Parameter Initialization Methods
================================

We can set up different initialization method for layer parameters.

.. code-block:: python

    from neupy import layers, algorithms, init

    gdnet = algorithms.GradientDescent(
        [
            layers.Input(10),
            layers.Sigmoid(30, weight=init.Normal()),
            layers.Sigmoid(15, weight=init.Normal()),
        ]
    )

Initialization class has an ability to generate parameters based on the specified shape. For instance, first sigmoid layer has 10 inputs and 30 outputs which means that this layer should have weight with shape ``(10, 30)``. During initialization we don't need to specify the shape of the parameter. This information would be provided to the Initializer class during layer initialization step.

It's possible to set up our own weight for layers. Let's do the same initialization procedure with manually generated weights.

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

Share parameters between layers
-------------------------------

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

It's very easy to create our own initialization method. All we need is just to inherit from the ``init.Initializer`` class and define ``sample`` method that accepts two argument (including the ``self`` argument).

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
            layers.Sigmoid(30, weight=Exponential(scale=0.02)),
            layers.Sigmoid(15, weight=Exponential(scale=0.05)),
        ]
    )
