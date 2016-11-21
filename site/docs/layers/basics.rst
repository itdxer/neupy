Basics
======

Layer is a building block for constructable neural networks. When network contains lots of layers it's become difficult to read architecture. With NeuPy it's not only easy to build deep learning models, but it's also easy to read network's architecture from the code.

Before we start I want to explain a few basics that we are going to use all the time.

Layers in the NeuPy can be defined independently from each other. Which means that we don't need to create first layer to be able to define a second one. Order matters only when we define relations between layers. The most useful function to define relations is ``layers.join``. It accepts sequence of layers and join them in the connection.

.. code-block:: python

    >>> from neupy import layers
    >>>
    >>> layers.join(layers.Sigmoid(1), layers.Sigmoid(2))
    Sigmoid(1) > Sigmoid(2)
    >>>
    >>> layers.join(layers.Sigmoid(2), layers.Sigmoid(1))
    Sigmoid(2) > Sigmoid(1)

Another way to define relations between layer is to use ``>`` operator.

.. code-block:: python

    >>> from neupy import layers
    >>>
    >>> connection = layers.Sigmoid(2) > layers.Sigmoid(1)
    >>> connection
    Sigmoid(2) > Sigmoid(1)

As you can see the syntax is very intuitive. One disadvantage is that it's difficult to construct networks that has more than 4 layers. This method is suitable only for small network. it's possible to use this syntax in deep networks in terms of subnetworks_ which we will discuss soon.

Input layer
-----------

Previous connections hasn't been completed yet.

.. code-block:: python

    >>> from neupy import layers
    >>>
    >>> connection = layers.Sigmoid(2) > layers.Sigmoid(1)
    >>> connection
    Sigmoid(2) > Sigmoid(1)
    >>>
    >>> print(connection.input_shape)
    None
    >>> connection.output_shape
    (1,)

Even thought we know the output shape we don't know an input. To be able to construct a full connection any network should have :layer:`Input` layer.

.. code-block:: python

    >>> connection = layers.Input(3) > connection
    >>> connection
    Input(3) > Sigmoid(2) > Sigmoid(1)
    >>>
    >>> connection.input_shape
    (3,)

Layer initialization
--------------------

Since layers are defined independently from each other we cannot perfom all initialization procedure after we connected layers. To be able to do that we need to call ``initialization`` method when all connections are defined.

.. code-block:: python

    >>> from neupy import layers
    >>>
    >>> sigmoid_layer = layers.Sigmoid(3)
    >>> connection = layers.Input(2) > sigmoid_layer
    >>>
    >>> sigmoid_layer.weight.get_value()
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    neupy.init.UninitializedException: Cannot get parameter value.
    Parameter hasn't been initialized yet
    >>>
    >>> connection.initialize()
    >>> sigmoid_layer.weight.get_value()
    array([[ 0.90131086,  0.38221973, -0.69804142],
           [-0.54882893,  0.81503922, -0.53348505]])

Only after the initialization we was able to get parameter.

Mutlilayer Perceptron (MLP)
===========================

Let's start with some practical examples. Here is a networks that classify 28x28 digit images.

.. code-block:: python

    from neupy import layers

    feedforward = layers.join(
        layers.Input(784),  # 28 * 28 = 784
        layers.Relu(500),
        layers.Relu(300),
        layers.Softmax(10),
    )

.. figure:: images/feedforward-graph-connection.png
    :align: center
    :alt: Feedforward connections in NeuPy

You can see from the figure above that we have dense connection even though we didn't define them. In NeuPy you can define dense connections within layers with activation function for simplicity. We can split layer with activation functions into simplier operations.

.. code-block:: python

    from neupy import layers

    connection = layers.join(
        layers.Input(784),

        layers.Linear(500),
        layers.Relu(),

        layers.Linear(300),
        layers.Relu(),

        layers.Linear(10),
        layers.Softmax(),
    )

This connection has exactly the same structure as the previous one. We just split each layer with activation function into simple operations. Operation in the ``layers.Relu(500)`` is equivalent to ``layers.Linear(500) > layers.Relu()``.

Convolutional Neural Networks (CNN)
===================================

NeuPy supports Convolutional Neural Networks. Here is an example of simple CNN.

.. code-block:: python

    from neupy import layers

    convnet = layers.join(
        layers.Input((3, 28, 28)),

        layers.Convolution((32, 3, 3)),
        layers.Relu(),
        layers.Convolution((48, 3, 3)),
        layers.Relu(),
        layers.MaxPooling((2, 2)),

        layers.Reshape(),
        layers.Softmax(10),
    )

.. figure:: images/conv-graph-connection.png
    :align: center
    :alt: Feedforward convolutional connections in NeuPy

Convolution
-----------

.. code-block:: python

    layers.Convolution((32, 3, 3))

NeuPy supports only 2D convolution. It's trivial to make a 1D convoltion. You just need to set up width or height equal to 1. In this section I will focuse only on 2D convoltuin, but with a small modifications everything work for 1D as well.

In the previous example you should see a network that contains a couple of convolutional layers. Each of these layers takes one mandatory argument that defines convolutional layer structure. Each parameter excepts a tuple that contains three integers ``(output channels, filter rows, filter columns)``. Information about the input channel takes from the previous layer.

Convolutional layer has a few other attributes that you can modify. You can check the :layer:`Convolutional <Convolution>` layer's documentation and find more information about this layer type.

Pooling
-------

.. code-block:: python

    layers.MaxPooling((2, 2))

Pooling works very similar. As in the convolutional layer you also need to set up one mandatory attribute as a tuple. But in case of pooling layer this attribute should be a tuple that contains only two integers. This parameters defines a factor by which to downscale ``(vertical, horizontal)``. (2, 2) will halve the image in each dimension.

Pooling defined as for the 2D layers, but you also can use in case of 1D convolution. In that case you need to define one of the downscale factors equal to 1. For instance, it can be somethig like that ``(1, 2)``.

Reshape
-------

.. code-block:: python

    layers.Reshape()

This layer basically do the same as `numpy.reshape <https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html>`_ function. One different is that the shape argument is optional. When shape is not defined :layer:`Reshape` layer converts input to 2D matrix.

.. code-block:: python

    >>> from neupy import layers
    >>> connection = layers.Input((3, 10, 10)) > layers.Reshape()
    >>> connection.input_shape
    (3, 10, 10)
    >>> connection.output_shape
    (300,)

Additional argument for the :layer:`Reshape` layer can help to define a new shape for the input tensor.

.. code-block:: python

    >>> from neupy import layers
    >>> connection = layers.Input((3, 10, 10)) > layers.Reshape((3, 100))
    >>> connection.input_shape
    (3, 10, 10)
    >>> connection.output_shape
    (3, 100)

.. raw:: html

    <br>

Graph connections
=================

Any connection between layers in NeuPy is a `Directional Acyclic Graph (DAG) <https://en.wikipedia.org/wiki/Directed_acyclic_graph>`_. So far we've encountered only sequential connections. Here is an example of network with parallel connections.

.. code-block:: python

    from neupy import layers

    connection = layers.join(
        layers.Input((3, 10, 10)),
        layers.Parallel(
            [[
                layers.Convolution((32, 3, 3)),
                layers.Relu(),
                layers.MaxPooling((2, 2)),
            ], [
                layers.Convolution((16, 7, 7)),
                layers.Relu(),
            ]],
            layers.Concatenate()
        ),
        layers.Reshape(),
        layers.Softmax(10),
    )

.. figure:: images/conv-parallel-connection.png
    :align: center
    :alt: Graph connections in NeuPy

You can see two new layers. The first one is the :layer:`Parallel` layer. This layer accepts two parameters. First one is an array of multiple connections. As you can see from the figure above each of the connections above accepts the same input, but each of the do different transformation to this input. The second parameter is an layer that accepts multiple inputs and combine then into single output. From our example we can see that from the left branch we got output shape equal to ``(32, 4, 4)`` and from the right branch - ``(16, 4, 4)``. The :layer:`Concatenate` layer joins layers over the firts dimension and as output returns tensor with shape ``(48, 4, 4)```

.. raw:: html

    <br>

.. _subnetworks:

Subnetworks
===========

Subnetworks is a simple trick that makes easier to read and understend network's structure. Instead of explaining it's much easier to show the main advantage of this method. Here is an example of the simpe convolutional network.

.. code-block:: python

    from neupy import layers

    connection = layers.join(
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

        layers.Relu(1024),
        layers.BatchNorm(),

        layers.Softmax(10),
    )

Does it look simple to you? Not at all. However, this is a really simple network. It looks a bit complecated because it contains a lot of simple layers that usually different libraries combine in one. For instance, non-linearity like :layer:`Relu` is usually built-in inside the :layer:`Convolution` layer. So instead of combining simple layers in one complecated in NeuPy it's better to use subnetworks. Here is an example on how to re-write network's structure from the previous example in terms of subnetworks.

.. code-block:: python

    from neupy import layers

    connection = layers.join(
        layers.Input((1, 28, 28)),

        layers.Convolution((32, 3, 3)) > layers.Relu() > layers.BatchNorm(),
        layers.Convolution((48, 3, 3)) > layers.Relu() > layers.BatchNorm(),
        layers.MaxPooling((2, 2)),

        layers.Convolution((64, 3, 3)) > layers.Relu() > layers.BatchNorm(),
        layers.MaxPooling((2, 2)),

        layers.Reshape(),

        layers.Relu(1024) > layers.BatchNorm(),
        layers.Softmax(10),
    )

As you can see we use an ability to organize sequence of simple layer in one small network. Each subnetwork defines a sequence of simple operations. You can think about subnetworks as a simple way to define more complecated layers. But instead of creating redundant classes that define complex layers you can define everything in place. In addition it improves the readability, because now you can see order of these simple operations inside the subnetwork.
