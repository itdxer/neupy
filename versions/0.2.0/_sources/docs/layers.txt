Layers
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

Types
*****

There are two main types of layers.
First type includes layers that have weights and activation function.
The second one is output layers.
Output layer is always the final layer in network structure and it just makes final output transformation for neural network.
The output layer doesn't have weights or activation function and it doesn't
involve in training procedure. It's is just a useful feature that helps to make
finall transformations with neural network's output.

Create custom layers
********************

The simplest type of two layers types is an output layer. Below you can see simple example.

.. code-block:: python

    from neupy import layers

    class RoundFloorOutput(layers.Output):
        def output(self, value):
            return value.astype(int)

The base class is :layer:`Output`.
This layer has one useful method - ``output``.
Attribute ``value`` contains the raw output from network and method can manage to perform some useful transformation to provide different output.

Other layers should have activation function.
The example below shows one of the possible way to create a new layer.

.. code-block:: python

    import theano.tensor as T
    from neupy import layers

    class SquareLayer(layers.Layer):
        activation_function = T.square

First of all you can see different class :layer:`Layer`.
This class expect ``activation_function`` property to be provided. It must be
an one-argument function that returns Theano function.
In this example we just use simple function which squares input value.

And a low-level implementation of layer inherits :layer:`BaseLayer` class and contains method ``output``.
It can be useful if you want to create a layer which will have custom behaviour.

.. code-block:: python

    from neupy import layers

    class PlusOneLayer(layers.BaseLayer):
        def output(self, input_value):
            return input_value + 1
