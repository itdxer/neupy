Basics
------

In this chapter you will see how to set up different connections for the neural network.

Syntax
******

There are three ways to set up your connection between layers. First one is the simplest one. You just define a list or tuple with the integers. Each integer in the sequence identifies layer's size.

.. code-block:: python

    from neupy import algorithms
    bpnet = algorithms.GradientDescent((2, 4, 1))

In that way we don't actually set up any layer types. By default NeuPy constructs from the tuple simple MLP networks that contains dense layers with sigmoid as a nonlinear activation function.

The second method is the most common one.

.. code-block:: python

    from neupy import algorithms, layers

    bpnet = algorithms.GradientDescent(
        [
            layers.Input(10),
            layers.Sigmoid(40),
            layers.Sigmoid(2)
            layers.Softmax(2),
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
        Input(10) > Sigmoid(40) > Sigmoid(2),
        step=0.2,
        shuffle_data=True
    )

This one is not very useful in case when you have more than 5 layers. This method can be more useful in case of subnetworks.

Layers with activation function
*******************************

This layer can have two different behaviours. You've already seen the first case in the previous chapter. Let's show it again.

.. code-block:: python

    from neupy import algorithms, layers

    bpnet = algorithms.GradientDescent(
        [
            layers.Input(10),
            layers.Sigmoid(40),
            layers.Sigmoid(2)
            layers.Softmax(2),
        ],
        step=0.2,
        shuffle_data=True
    )

Let's consider the ``layers.Sigmoid(40)`` layer. This layer defines two different operations. The first one is a linear transformation that multiplies input by a weight matrix and adds bias to the result of this multiplication. The second one applies activation function to the linear transformation. The type of the linear function depends on the layer class. In the examples we use two layers with sigmoid activation function and one with a softmax.

This layer combines two basic operations, but you can split it into the separate steps. Here is a previous examples divided into the separate steps.

.. code-block:: python

    from neupy import algorithms, layers

    bpnet = algorithms.GradientDescent(
        [
            layers.Input(10),

            layers.Linear(40),
            layers.Sigmoid(),

            layers.Linear(2),
            layers.Sigmoid(),

            layers.Linear(2),
            layers.Softmax(),
        ],
        step=0.2,
        shuffle_data=True
    )

This structure defines exactly the same structure as in the first example, but it takes more layers to define it. In case of MLP networks the first example is easier to read. The main advantage of it is a readability. You can separate stacked layers in to the two columns. This first one defines the order of activation functions. In our examples it is ``sigmoid > sigmoid > softmax``. And the second column shows you the network's structure in the different layers. In out example it is ``10 > 40 > 2 > 2``.

And probably you've noticed the other way to use layer with activation function. If you don't set up layer's output size it will not apply linear transformation procedure and just will pass input value through the nonlinear activation function.

Subnetworks
***********

Subnetworks is a simple trick that makes easier to read and understend network's structure. Instead of explaining it's much easier to show the main advantage of this method. Here is an example of the simpe convolutional network.

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

            layers.Relu(1024),
            layers.BatchNorm(),

            layers.Softmax(10),
        ]
    )

Does it look simple to you? Not at all. However, this is a really simple network. It looks a bit complecated because it contains a lot of simple layers that usually different libraries combine in one. For instance, non-linearity like :layer:`Relu` is usually built-in inside the :layer:`Convolution` layer. So instead of combining simple layers in one complecated in NeuPy it's better to use subnetworks. Here is an example on how to re-write network's structure from the previous example in terms of subnetworks.

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

            layers.Relu(1024) > layers.BatchNorm(),
            layers.Softmax(10),
        ]
    )

As you can see we use an ability to organize sequence of simple layer in one small network. Each subnetwork defines a sequence of simple operations. You can think about subnetworks as a simple way to define more complecated layers. But instead of creating redundant classes that define complex layers you can define everything in place. In addition it improves the readability, because now you can see order of these simple operations inside the subnetwork.
