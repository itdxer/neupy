.. _subnetworks:

Subnetworks
===========

**Subnetworks** is a method that improves readability of the neural network architecture. Instead of explaining, it's much easier to show the main advantage of this technique. Here is an example of the simple convolutional network.

.. code-block:: python

    from neupy.layers import *

    network = layers.join(
        Input((1, 28, 28)),

        Convolution((32, 3, 3)),
        Relu(),
        BatchNorm(),

        Convolution((48, 3, 3)),
        Relu(),
        BatchNorm(),
        MaxPooling((2, 2)),

        Convolution((64, 3, 3)),
        Relu(),
        BatchNorm(),
        MaxPooling((2, 2)),

        Reshape(),

        Relu(1024),
        BatchNorm(),

        Softmax(10),
    )

Does it look simple to you? Most likely not. However, this is a really simple neural network. It looks a bit complicated, because it contains a lot of simple layers that usually combined into one. For instance, non-linearity like :layer:`Relu` is usually built-in inside the :layer:`Convolution` layer. So instead of combining simple layers in one complicated, in NeuPy it's better to use subnetworks. Here is an example on how to re-write network's structure from the previous example in terms of subnetworks.

.. code-block:: python

    network = layers.join(
        Input((28, 28, 1)),

        Convolution((3, 3, 32)) >> Relu() >> BatchNorm(),
        Convolution((3, 3, 48)) >> Relu() >> BatchNorm(),
        MaxPooling((2, 2)),

        Convolution((3, 3, 64)) >> Relu() >> BatchNorm(),
        MaxPooling((2, 2)),

        Reshape(),

        Relu(1024) >> BatchNorm(),
        Softmax(10),
    )

As you can see, we use an ability to organize sequence of simple layer in one small network. Each subnetwork defines a sequence of simple operations. You can think about subnetworks as a simple way to define more complicated layers. But instead of creating redundant classes or functions, that define complex layers, we can define everything in place. In addition, it improves the readability, because now everybody can see order of these simple operations inside the subnetwork.
