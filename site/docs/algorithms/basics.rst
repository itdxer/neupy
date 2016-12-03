Basics
======

NeuPy supports many different neural network types. These networks we can divide into two major categories.

1. Neural Networks with constructible architecture.
2. Neural Networks with fixed architecture.

Networks with constructible architecture is basically a field related to the deep learning. These algorithms allows us to construct different neural network architectures. Another type is networks with fixed architecture. There are many different classes like :network:`RBM`, :network:`PNN`, :network:`CMAC` and so on. These networks has pre-defined architecture and there is no way to add or remove layers from it, but they have parameters that allow us to control them.

Despite the major differences between neural network algorithms the API for them looks very similar.

Initialization
--------------

This is a first step to initialize neural network. Each algorithms has it's own set of parameters. Let's check :network:`MinibatchGradientDescent` algorithm as an example. Since most of the algorithms has lots of different parameters, NeuPy doesn't support ordered arguments.

.. code-block:: python

    from neupy import algorithms

    # Invalid initialization
    gdnet = algorithms.MinibatchGradientDescent((2, 3, 1), 16, 'mae')

    # Valid initialization
    gdnet = algorithms.MinibatchGradientDescent(
        (2, 3, 1),
        batch_size=16,
        error='mae'
    )

Training
--------

To train neural network we need to use ``train`` method.

.. code-block:: python

    from neupy import algorithms

    nnet = algorithms.MinibatchGradientDescent((2, 3, 1))
    nnet.train(x_train, y_train, epochs=1000)

If we need to validate our training results with validation dataset we can pass it as an additional argument.

.. code-block:: python

    from neupy import algorithms

    nnet = algorithms.MinibatchGradientDescent((2, 3, 1))
    nnet.train(x_train, y_train, x_test, y_test, epochs=1000)

To be able to see output after each epoch we can set up ``verbose=True`` in the network initialization step.

.. code-block:: python

    from neupy import algorithms
    nnet = algorithms.MinibatchGradientDescent((2, 3, 1), verbose=True)

Or we can switch on ``verbose`` parameter after the initialization.

.. code-block:: python

    from neupy import algorithms

    nnet = algorithms.MinibatchGradientDescent((2, 3, 1))

    nnet.verbose = True
    nnet.train(x_train, y_train, x_test, y_test, epochs=1000)

Prediction
----------

To make a prediction we need to pass dataset to the ``predict`` method.

.. code-block:: python

    y_predicted = nnet.predict(x_test)
