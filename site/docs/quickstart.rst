Quick start
===========

This guide provides basic overview of the NeuPy library.

Loading data
------------

Building model
--------------

.. code-block:: python

    from neupy.layers import *
    network = Input(24) > Relu(12) > Softmax(10)

Inline connection is a suitable choice for very small networks. Large networks can be defined with the help of the `join` operator.

.. code-block:: python

    from neupy import layers

    network = layers.join(
        layers.Input(30),
        layers.Relu(24),
        layers.Softmax(10),
    )

Inline connections can also improve readability in the large networks when only some groups of layers defined with the help of this operator. See :doc:`Subnetworks <layers/basics#subnetworks>` in order to learn more.

Training
--------

.. code-block:: python

    from neupy import algorithms

    optimizer = algorithms.Momentum(network, step=0.1)
    optimizer.train(x_train, y_train, x_test, y_test, epochs=100)

Evaluation
----------

.. code-block:: python

    y_predicted = optimizer.predict(x_test)

What's next?
------------

There are available a few more tutorials that can help you to start working with NeuPy.

Additional information about the library you can find in the documentation.
