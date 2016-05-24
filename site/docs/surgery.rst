Neural Network Surgery
======================

It's very easy to define relations between layers in NeuPy. Also NeuPy's syntax makes it easy to read and understand your network's strucutre. But any method should have its pros and cons. The main disadvantage of this approach is that it's a bit complecated to use a part of the network when you define full network. To solve this issue in NeuPy exists module named as a surgery. The idea is simple. You still can define your networks using very simple syntax and then cut pieces from them and connect these pieces together in the new networks.

Cutting layers from the network
*******************************

Cut is a first feature of a surgery module. You can cut network in a few different ways. To make it simple to understand I'm going to cinsider a simple pretraining autoencoder problem. Let's define the autoencoder network.

.. code-block:: python

    from neupy import algorithms, layers

    autoencoder = algorithms.Momentum(
        [
            layers.Input(784),
            layers.Sigmoid(100),
            layers.Sigmoid(784),
        ],
        step=0.1,
        momentum=0.99,
        verbose=True,
        error='rmse',
    )

    x_train, y_train = load_mnist()
    autoencoder.train(x_train, x_train, epochs=20)

As you can see we define a very simple autoencoder that trains over 20 epochs. Trained networks contain two pieces that usually known as encoder and decoder networks. In case of pretraining we are interested in encoding layer. Let's cut this part.

.. code-block:: python

    >>> from neupy import surgery
    >>> encoder = surgery.cut(autoencoder, start=0, end=2)
    >>> encoder
    Input(784) > Sigmoid(100)

As you can see this operation is similar to Python's slicings. Basically this operation do something like this.

.. code-block:: python

    # NOTE: This is a pseudo-code, so it will
    # not work in the NeuPy
    encoder = autoencoder_layers[0:2]

Since we can part of the network that do an encoding procedure we can attach pretrained part to the other network that will manage to do classification.

.. code-block:: python

    classifier = algorithms.Momentum(
        encoder > layers.Softmax(10),
        step=0.1,
        momentum=0.99,
        verbose=True,
        error='categorical_crossentropy',
    )
    classifier.train(x_train, y_train, epochs=10)

That's it, now you have classifier with pretrained layer. Now you can check its final structure.

.. code-block:: python

    >>> classifier.architecture()
    -----------------------------------------------
    | # | Input shape | Layer Type | Output shape |
    -----------------------------------------------
    | 1 | 784         | Input      | 784          |
    | 2 | 784         | Sigmoid    | 100          |
    | 3 | 100         | Softmax    | 10           |
    -----------------------------------------------

Such a method can be unsutable when you deal with networks that have more than 10 layers. To make simplify these procedure for the bigger networks NeuPy gives an ability to predefine places where you need to cut network into pieces. Surgery module contains class ``CutLine``. These class defines places where you want to cut network. Let's consider another example. Suppose we need to cut all hidden layers from the network. Here is an example on how we can do that with predefined layout.

.. code-block:: python

    from neupy import algorithms, layers, surgery
    network = algorithms.GradientDescent([
        layers.Input(5),

        surgery.CutLine(),  # <- first cut point

        layers.Sigmoid(10),
        layers.Sigmoid(20),
        layers.Sigmoid(30),

        surgery.CutLine(),  # <- second cut point

        layers.Sigmoid(1),
    ])

In the surgery module there exists another function that can do this procedure. Here is how it works.

.. code-block:: python

    >>> cutted_connections = surgery.cut_along_lines(network)
    >>>
    >>> for connection in cutted_connections:
    ...     print(connection)
    ...
    Input(5)
    Sigmoid(10) > Sigmoid(20) > Sigmoid(30)
    Sigmoid(1)

It returns a list that contains all of these pieces. Now you can get piece that yu are interested in.

.. code-block:: python

    >>> _, hidden_layers, _ = cutted_connections
    >>> hidden_layers
    Sigmoid(10) > Sigmoid(20) > Sigmoid(30)

As in the autoencoder case we can use this layers in the other networks

Sew layers together
*******************

Surgery module not only can break networks into pieces, but it also cut join different pieces together. It's known as **sewing**. We can use cutted layers from the previous example.

.. code-block:: python

    >>> encoder
    Input(784) > Sigmoid(100)
    >>>>
    >>> hidden_layers
    Sigmoid(10) > Sigmoid(20) > Sigmoid(30)

Let's imagine that we need to join them together. The main problem is that we are not able to combine networks together. The reason is that the ``encoder`` produces 100 dimensional output, while ``hidden_layers`` expects 5 dimensional input. To connect them together we need to define intermidiate layer.

.. code-block:: python

    >>> connected_layers = surgery.sew_together([
    ...     encoder,
    ...     layers.Relu(5),
    ...     hidden_layers
    ... ])
    >>> connected_layers
    Input(784) > Sigmoid(100) > Relu(5) > Sigmoid(10) > Sigmoid(20) > Sigmoid(30)
