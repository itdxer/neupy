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

Initialization class has an ability to generate parameters based on the specified shape. For instance, a first sigmoid layer has 10 inputs and 30 outputs which mean that this layer should have weight with shape ``(10, 30)``. During initialization, we don't need to specify the shape of the parameter. This information would be provided to the Initializer class during layer initialization step.

It's possible to set up any value for the weight as long as it has valid shape. Let's do the same initialization procedure with manually generated weights.

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

Code above does the same type of the initialization as in the previous example, the only problem that we need to hard-code expected shape of the weights.

More initialization methods you can find :ref:`here <init-methods>`.

Create custom initialization methods
------------------------------------

It's very easy to create custom initialization method. All we need is just to inherit from the ``init.Initializer`` class and define ``sample`` method that accepts one argument (excluding the ``self`` argument). Argument will contain shape of the output tensor that we expect to get.

In the example below we create custom initializer that samples weights from the exponential distribution.

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

Share parameters between layers
-------------------------------

In some applications it might be useful to share parameters from other layers. In the example below, we initialize two hidden layers and each layer has exactly the same weight and bias.

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

It works in the following way. First, we connected input layer to the hidden layer. This operation triggered parameter initialization for the first hidden layer.

.. code-block:: python

    >>> hidden_layer_1 = layers.Relu(10)
    >>> network = layers.Input(10) > hidden_layer_1

Next, we associated parameters from the first hidden layer with parameters in the second layer.

.. code-block:: python

    >>> hidden_layer_2 = layers.Relu(10, weight=hidden_layer_1.weight,
    ...                              bias=hidden_layer_1.bias)

Notice that in this case weight and bias are instance of the ``Variable`` class from the Tensorflow.

.. code-block:: python

    >>> hidden_layer_1.weight
    <tf.Variable 'layer/relu-1/weight:0' shape=(10, 10) dtype=float32_ref>
