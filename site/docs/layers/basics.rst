Basics
------

Syntax
******

In the quick start chapter we saw the neural network that contains only sigmoid
layers which are default in :network:`GradientDescent` algorithms.
But many problems need the specific structure for neural network connections.
In this chapter you will see how to set up different connections for the neural network.

There are three ways to set up your neural network connection.
First one is the simplest one.
You just define a list or tuple with the numbers of units for each layer in accordance to their order.

.. code-block:: python

    from neupy import algorithms
    bpnet = algorithms.GradientDescent((2, 4, 1))

The second method is the most useful for tasks when you just want to test your network
structure and don't create final one for it.
For example it can look like this.

.. code-block:: python

    from neupy import algorithms, layers

    bpnet = algorithms.GradientDescent(
        [
            layers.Sigmoid(10),
            layers.Sigmoid(40),
            layers.Softmax(2),
            layers.Output(2)
        ],
        step=0.2,
        shuffle_data=True
    )

And the last one is the most intuitive.

.. code-block:: python

    from neupy import algorithms
    from neupy.layers import *

    bpnet = algorithms.GradientDescent(
        Sigmoid(10) > Sigmoid(40) > Output(2),
        step=0.2,
        shuffle_data=True
    )
