Quick start
===========

This guide provides basic overview of the NeuPy library. For more detailed examples, see articles and examples available in the `tutorials <http://neupy.com/docs/tutorials.html>`_ page.

Building model
--------------

NeuPy provide very simple and intuitive interface for building neural networks. Simple architecture can be defined with a help of the inline operator.

.. code-block:: python

    from neupy.layers import *
    network = Input(24) >> Relu(12) >> Softmax(10)

Inline connection is a suitable choice for very small networks. Large networks can be defined with the help of the ``join`` operator.

.. code-block:: python

    network = join(
        Input(24),
        Relu(12),
        Softmax(10),
    )

Inline connections can also improve readability in the large networks when only some groups of layers were defined with the help of this operator. See `subnetworks <http://neupy.com/docs/layers/basics.html#subnetworks>`_ topic in order to learn more.

Training
--------

Training can be done in two simple steps. First, we need to specify training algorithm. And second, we need to pass training data and specify number of training epochs. It can be done in two lines of code in NeuPy.

.. code-block:: python

    from neupy import algorithms

    optimizer = algorithms.Momentum(network, step=0.1, verbose=True)
    optimizer.train(x_train, y_train, x_test, y_test, epochs=100)

Evaluation
----------

After the training, we can propagate test inputs through the network and get prediction.

.. code-block:: python

    y_predicted = optimizer.predict(x_test)

What's next?
------------

There are available a few more tutorials that can help you to start working with NeuPy. You can visit the `tutorials <http://neupy.com/docs/tutorials.html>`_ page or you can click on one of the links below.

* :ref:`mnist-classification`
* :ref:`boston-house-price`

Additional information about the library you can find in the `documentation <file:///Users/projects/neupy/site/blog/html/pages/documentation.html>`_.
