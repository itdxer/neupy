Basics
------

In this chapter you will see how to set up different connections for the neural network.

Syntax
******

There are three ways to set up your neural network connection between layers. First one is the simplest one. You just define a list or tuple with the integers. Each integer in the sequence identifies layer's size.

.. code-block:: python

    from neupy import algorithms
    bpnet = algorithms.GradientDescent((2, 4, 1))

In that way we don't actually set up any layer types. By default NeuPy constructs from the tuple simple MLP networks that contains dense layers with sigmoid as a nonlinear activation function.

The second method is the most useful common one. Here is a simple example.

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

The networks structure is clear from the example. You basically organize layer's sequence in on big list.

And the last one is the most intuitive.

.. code-block:: python

    from neupy import algorithms
    from neupy.layers import *

    bpnet = algorithms.GradientDescent(
        Sigmoid(10) > Sigmoid(40) > Output(2),
        step=0.2,
        shuffle_data=True
    )

This one is not very useful in case when you have more than 5 layers. This method can be more useful in case of subnetworks.

Subnetworks
***********

Subnetworks is a simple trick that makes easier to read and understend the networks structure. Instead of explaining it's much easier to show the main advantage of this method. Here is an example of the simpe convolutional network.

.. code-block:: python

    from neupy import algorithms, layers

    network = algorithms.Adadelta(
        [
            layers.Input((1, 28, 28)),

            layers.Convolution((32, 3, 3)),
            layers.Relu(),
            layers.BatchNorm(),

            layers.Convolution((48, 3, 3)),
            layers.Relu(),
            layers.BatchNorm(),
            layers.MaxPooling((2, 2)),

            layers.Convolution((64, 3, 3)),
            layers.Relu(),
            layers.BatchNorm(),
            layers.MaxPooling((2, 2)),

            layers.Reshape(),

            layers.Relu(64 * 5 * 5),
            layers.BatchNorm(),

            layers.Softmax(1024),
            layers.ArgmaxOutput(10),
        ]
    )

Does it look simple to you? Not at all. However, this is a really simple network. It looks a bit complecated because it contains a lot of simple layers that usually different libraries combine in the more complecated one. For instance, non-linearity like :layer:`Relu` is usually built-in inside the :layer:`Convolution` layer. NeuPy supports simplicity and in addition it improves the readability of your networks structure. That's why become useful subnetworks. Here is an example on how to re-write the network's structure from the previous example in the terms of subnetworks.

.. code-block:: python

    from neupy import algorithms, layers

    network = algorithms.Adadelta(
        [
            layers.Input((1, 28, 28)),

            layers.Convolution((32, 3, 3)) > layers.Relu() > layers.BatchNorm(),
            layers.Convolution((48, 3, 3)) > layers.Relu() > layers.BatchNorm(),
            layers.MaxPooling((2, 2)),

            layers.Convolution((64, 3, 3)) > layers.Relu() > layers.BatchNorm(),
            layers.MaxPooling((2, 2)),

            layers.Reshape(),

            layers.Relu(64 * 5 * 5) > layers.BatchNorm(),
            layers.Softmax(1024),
            layers.ArgmaxOutput(10),
        ]
    )

As you can see we use an ability to organize sequence of simple layer in one small network. Next we basically include this small networks in the sequence.
