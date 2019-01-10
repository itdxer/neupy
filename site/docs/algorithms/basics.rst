Basics
======

NeuPy supports many different neural network architectures. They can be divided into two categories.

1. Neural networks with constructible architecture.
2. Neural networks with fixed architecture.

As the name suggests, constructible architectures can be build using layers as building blocks and it has to be specified by the user. To another type belong networks with fixed architecture. There are many different networks like :network:`RBM`, :network:`PNN`, :network:`CMAC`. These networks have predefined architecture and there is no way to add or remove layers from it.

Initialization
--------------

Initialization process is slightly different for networks with fixed and constructible architectures. Networks with fixed architecture have predefined structure and we need just to specify algorithm-specific parameters.

.. code-block:: python

    from neupy import algorithms
    sofm = algorithms.SOFM(n_inputs=2, n_outputs=4)

Notice, that we explicitly specified argument names during initialization. NeuPy doesn't have defined order for the arguments and it will raise exception when there is no names associated with each argument.

.. code-block:: python

    # Invalid initialization, because every additional
    # argument require to have name specified explicitly
    sofm = algorithms.SOFM(2, 4)

In contrast, network with constructible architectures require two basic steps. First, you need to define structure of the network. For example:

.. code-block:: python

    from neupy import layers

    network = layers.join(
        layers.Input(10),
        layers.Relu(5),
        layers.Softmax(4),
    )

When architecture defined we need to specify training algorithm. The initialization looks exactly the same as the one that we've seen for networks with fixed architectures. The only difference is that we have to specify our network structure as the first argument.

.. code-block:: python

    from neupy import algorithms
    momentum = algorithms.Momentum(connection, alpha=0.1, nesterov=True)

These two steps can be combined into one for simplicity.

.. code-block:: python

    from neupy import layers, algorithms

    momentum = algorithms.Momentum(
        [
            layers.Input(10),
            layers.Relu(5),
            layers.Softmax(4),
        ],
        alpha=0.1,
        nesterov=True,
    )

Training
--------

Training looks the the same for all algorithms, with few exceptions for different algorithms, so you should refer to the documentation before training, in case you're not familiar with the API.

To train neural network we need to use the ``train`` method (or ``fit`` which was added for ``scikit-learn`` compatibility).

.. code-block:: python

    network.train(x_train, y_train, epochs=1000)

If we need to validate our training results with validation dataset we can pass it as an additional argument (that option available for most of the algorithms, but not all of them).

.. code-block:: python

    network.train(x_train, y_train, x_test, y_test, epochs=1000)

To be able to see the output after each epoch we can set up ``verbose=True`` during network's initialization.

.. code-block:: python

    from neupy import algorithms
    nnet = algorithms.Momentum(connection, verbose=True)

Or we can switch on ``verbose`` parameter after the initialization.

.. code-block:: python

    from neupy import algorithms

    nnet = algorithms.Momentum(connection, verbose=False)

    nnet.verbose = True
    nnet.train(x_train, y_train, x_test, y_test, epochs=1000)

If you want to run training in loop you have to change the way neupy outputs its training summary. It can be changed with the ``summary`` argument.


.. code-block:: python

    for _ in range(1000):
        nnet.train(x_train, y_train, epochs=1)

Prediction
----------

To make a prediction we need to pass networks input to the ``predict`` method.

.. code-block:: python

    y_predicted = nnet.predict(x_test)
