Parameter Initialization Methods
================================

In the NeuPy, initialization methods per layer can be be modified using classes from the ``init`` module.

.. code-block:: python

    from neupy.layers import *
    from neupy import init

    network = join(
        Input(10),
        Sigmoid(30, weight=init.Normal()),
        Sigmoid(15, weight=init.Normal()),
    )

Initialization class has an ability to generate parameters based on the specified shape. For instance, a first sigmoid layer expects 10 input features and generates 30 output features, which mean that this layer should have weight with shape ``(10, 30)``. During initialization, we don't need to specify the shape of the parameter. This information would be provided to the initializer class during weight initialization procedure.

It's possible to set up any value for the weight as long as it has valid shape. We can do the same initialization procedure with manually generated weights.

.. code-block:: python

    import numpy as np

    network = join(
        Input(10),
        Sigmoid(30, weight=np.random.randn(10, 30)),
        Sigmoid(15, weight=np.random.randn(30, 15)),
    )

Code above does the same type of the initialization as in the previous example, the only problem that we need to hard-code expected shape of the weights.

More initialization methods you can find :ref:`here <init-methods>`.

Create custom initialization methods
------------------------------------

It's very easy to create custom initialization method. All we need is just to inherit from the ``init.Initializer`` class and define ``sample`` method that accepts one argument (excluding the ``self`` argument). Argument will contain shape of the output tensor that we expect to get.

In the example below, we create custom initializer that samples weights from the exponential distribution.

.. code-block:: python

    import tensorflow as tf
    from neupy.layers import *
    from neupy import init

    class Gamma(init.Initializer):
        def __init__(self, alpha=0.01):
            self.alpha = alpha

        def sample(self, shape):
            return tf.random.gamma(shape, self.alpha)

    network = join(
        Input(10),
        Sigmoid(30, weight=Gamma(alpha=0.02)),
        Sigmoid(15, weight=Gamma(alpha=0.05)),
    )

Notice that the ``sample`` method returns Tensorflow's tensor. It's possible to return numpy's array, but in this case initialization might take more time, since we will need to generate weights per each variable sequentially.

Share parameters between layers
-------------------------------

In some applications, it might be useful to share parameters some of the between layers. Tensorflow's variable can be created in the code and passed to the layers that has to have shared parameters.

.. code-block:: python

    >>> import tensorflow as tf
    >>> from neupy.layers import *
    >>>
    >>> shared_weight = tf.Variable(tf.ones((10, 10)), name='shared-weight')
    >>>
    >>> hid1 = Relu(10, weight=shared_weight)
    >>> hid2 = Relu(10, weight=shared_weight)
    >>>
    >>> network = Input(10) >> hid1 >> hid2
    >>> network
    (?, 10) -> [... 3 layers ...] -> (?, 10)

We can check that only 1 weight variables will be used during the training.

.. code-block:: python

    >>> [v.name for v in network.variables.values()]
    ['shared-weight:0', 'layer/relu-1/bias:0', 'layer/relu-2/bias:0']

As we can see, only one weight variables is used, but two different biases were created per each layer, since we didn't create separate variable for them.
