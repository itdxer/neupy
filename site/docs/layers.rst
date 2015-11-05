Layers
------

Syntax
******

In the quick start chapter we saw the neural network that contains only sigmoid
layers which are default in :network:`Backpropagation` algorithms.
But many problems need the specific structure for neural network connections.
In this chapter you will see how to set up different connections for the neural network.

There are three ways to set up your neural network connection.
First one is the simplest one.
You just define a list or tuple with the numbers of units for each layer in accordance to their order.

.. code-block:: python

    from neupy import algorithms
    bpnet = algorithms.Backpropagation((2, 4, 1))

The second method is the most useful for tasks when you just want to test your network
structure and don't create final one for it.
For example it can look like this.

.. code-block:: python

    from neupy import algorithms, layers

    bpnet = algorithms.Backpropagation(
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

    bpnet = algorithms.Backpropagation(
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
The output layer doesn't have weights or activation function.

Create custom layers
********************

The simplest type of two layers types is an output layer. Below you can see simple example.

.. code-block:: python

    from neupy import layers

    class RoundFloorOutput(layers.Output):
        def format_output(self, value):
            return value.astype(int)

The base class is :layer:`Output`.
This layer has one useful method - ``format_output``.
Attribute ``value`` contains the raw output from network and method can manage to perform some useful transformation to provide different output.

Other layers should have activation function.
The example below shows one of the possible way to create a new layer.

.. code-block:: python

    from neupy import layers

    def square(x):
        return x ** 2

    class SquareLayer(layers.Layer):
        activation_function = square

First of all you can see different class :layer:`Layer`.
This class expect ``activation_function`` property to be provided that must be an one-argument function.
In this example we just use simple function which squares input value.

But we still can't use it in :network:`Backpropagation` algorithm because we don't describe derivative function.

.. code-block:: python

    from neupy import layers
    from neupy.functions import with_derivative

    def square_deriv(x):
        return 2 * x

    @with_derivative(square_deriv)
    def square(x):
        return x ** 2

    class SquareLayer(layers.Layer):
        activation_function = square


Now we can use it in :network:`Backpropagation` algorithm.
Also we can describe derivative for ``square_deriv`` function.

There also exist possibility to configure activation function.
Using the same example of square function we can make some general case of it.

.. code-block:: python

    from neupy import layers
    from neupy.core.properties import DictProperty
    from neupy.functions import with_derivative

    def square_deriv(x, a=1, b=0, c=0):
        return 2 * a * x + b

    @with_derivative(square_deriv)
    def square(x, a=1, b=0, c=0):
        return a * x ** 2 + b * x + c

    class SquareLayer(layers.Layer):
        function_coef = DictProperty(default={'a': 1, 'b': 0, 'c': 0})
        activation_function = square

    input_layer = SquareLayer(2, function_coef={'a': 1, 'b': 2, 'c': 3})

It's important for you to use the same number of constants in all derivative function even if they are disappear after differentiation.

And a low-level implementation of layer inherits :layer:`BaseLayer` class and contains method ``output``.
It can be useful if you want to create a layer which will have custom behaviour.

.. code-block:: python

    from neupy import layers

    class PlusOneLayer(layers.BaseLayer):
        def output(self, input_value):
            return input_value + 1
