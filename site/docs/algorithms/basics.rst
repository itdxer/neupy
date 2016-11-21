Basics
======

Initialization
--------------



Training
--------

When we defined our training algorithm we can train the network.

.. code-block:: python

    from neupy import algorithms

    nnet = algorithms.GradientDescent((2, 3, 1))
    nnet.train(x_train, y_train, epochs=1000)

If you want to validate your training results with validation dataset you can pass it as additional argument.

.. code-block:: python

    from neupy import algorithms

    nnet = algorithms.GradientDescent((2, 3, 1))
    nnet.train(x_train, y_train, x_test, y_test, epochs=1000)

To be able to see output after each epoch you can set up ``verbose=True`` in the network initialization step.

.. code-block:: python

    from neupy import algorithms
    nnet = algorithms.GradientDescent((2, 3, 1), verbose=True)

Or you can switch ``verbose`` mode after the initialization

.. code-block:: python

    nnet.vebose = True

Prediction
----------

After the training you can make a prediction.

.. code-block:: python

    y_predicted = nnet.predict(x_test)
