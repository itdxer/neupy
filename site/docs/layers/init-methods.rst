Parameter Initialization Methods
--------------------------------

This is a small topic that help you to understand how to initialize weight, bias and other parameters in NeuPy. The simplest way is to define initial weights inside the code.

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

This way is fine, but you can see that we need to hardcode parameter shape. In case if we want to change number of units in the hidden layer we need to update weight's shape as well. For more complicated rules code become messy. There is a better way to define parameters in the NeuPy. Here is an example.

.. code-block:: python

    from neupy import layers, algorithms
    from neupy.layers import init

    gdnet = algorithms.GradientDescent(
        [
            layers.Input(10),
            layers.Sigmoid(30, weight=init.HeNormal()),
            layers.Sigmoid(15, weight=init.HeNormal()),
        ]
    )

Basically we set up parameter equal to initialization method. So weight parameter will be defined based on the algorithm inside of the class.

More initialization methods you can find :ref:`here <init-methods>`.

Custom methods
**************

It is possible to define custom initialization method.

.. code-block:: python

    import numpy as np
    from neupy.layers import init
    from neupy import layers, algorithms

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
