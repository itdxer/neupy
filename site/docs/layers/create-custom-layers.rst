Custom layers
=============

There are a few main types of layers. We are going to check them one by one in order of complexity.

Element-wise transformation
***************************

In case

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

This layer has one useful method - ``output``.
Attribute ``value`` contains the raw output from network and method can manage to perform some useful transformation to provide different output.

Other layers should have activation function.
The example below shows one of the possible way to create a new layer.

.. code-block:: python

    import theano.tensor as T
    from neupy import layers

    class SquareLayer(layers.ActivationLayer):
        activation_function = T.square

First of all you can see different class :layer:`ActivationLayer`.
This class expect ``activation_function`` property to be provided. It must be
an one-argument function that returns Theano function.
In this example we just use simple function which squares input value.

And a low-level implementation of layer inherits :layer:`BaseLayer` class and contains method ``output``.
It can be useful if you want to create a layer which will have custom behaviour.

.. code-block:: python

    from neupy import layers

    class IncreaseByOneLayer(layers.BaseLayer):
        def output(self, input_value):
            return input_value + 1
