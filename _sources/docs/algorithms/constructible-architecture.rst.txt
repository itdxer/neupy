.. _constructible-architecture:

Algorithms with constructible architecture
==========================================

.. contents::

Specify network structure
-------------------------

There are two ways to define relations between layers. We can define network's architecture separately from the training algorithm.

.. code-block:: python

    from neupy import algorithms
    from neupy.layers import *

    network = Input(10) >> Sigmoid(40) >> Softmax(4)
    optimizer = algorithms.GradientDescent(
        network, step=0.2, shuffle_data=True, verbose=True,
    )

Or, we can set up a list of layers that define sequential relations between layers.

.. code-block:: python

    from neupy import algorithms

    optimizer = algorithms.GradientDescent(
        [
            Input(10),
            Sigmoid(40),
            Softmax(4),
        ],
        step=0.2,
        shuffle_data=True
        verbose=True,
    )

This is just a syntax simplification that allows to avoid using ``join`` function and inline connections.

Train networks with multiple inputs
-----------------------------------

NeuPy allows to train networks with multiple inputs.

.. code-block:: python

    from neupy import algorithms
    from neupy.layers import *

    optimizer = algorithms.GradientDescent(
        [
            parallel([
                # 3 categorical inputs
                Input(3),
                Embedding(n_unique_categories, 4),
                Reshape(),
            ], [
                # 17 numerical inputs
                Input(17),
            ]),
            Concatenate(),
            Relu(16),
            Sigmoid(1),
        ],

        step=0.5,
        verbose=True,
        loss='binary_crossentropy',
    )

    x_train_cat, x_train_num, y_train = load_train_data()
    x_test_cat, x_test_num, y_test = load_test_data()

    # Categorical variable should be the first, because
    # categorical input layer was defined first in the network
    optimizer.train(
        [x_train_cat, x_train_num], y_train,
        [x_test_cat, x_test_num], y_test,
        epochs=180,
    )
    y_predicted = optimizer.predict(x_test_cat, x_test_num)

Network in the example above has two inputs. Order of the inputs is important since first specified layer in the network will correspond to the first networks input. It's tru for the ``train``, ``score`` and ``predict`` methods

.. code-block:: python

    optimizer.train(
        [x_train_cat, x_train_num], y_train,
        [x_test_cat, x_test_num], y_test,
        epochs=180,
    )
    loss = optimizer.score([x_test_cat, x_test_num], y_test)
    y_predicted = optimizer.predict(x_test_cat, x_test_num)

Notice that ``predict`` method expects multiple inputs, unlike ``score`` and ``train`` methods. It's because for other methods it's important to differentiate between inputs and targets.

Algorithms
----------

NeuPy supports lots of different training algorithms based on the backpropagation. You can check :ref:`Cheat sheet <cheatsheet-backprop-algorithms>` if you want to learn more about them.

Before using these algorithms you must understand that not all of them are suitable for all problems. Some of the methods like :network:`Levenberg-Marquardt <LevenbergMarquardt>` or :network:`Conjugate Gradient <ConjugateGradient>` work better for small networks and they would be extremely slow for networks with millions parameters. In addition, it's important to note that not all algorithms are possible to train with mini-batches. Algorithms like :network:`Conjugate Gradient <ConjugateGradient>` don't work with mini-batches.

Loss functions
--------------

NeuPy has many different loss functions. These loss functions can be specified specified as a string.

.. code-block:: python

    from neupy import algorithms, layers

    nnet = algorithms.GradientDescent(
        [
            layers.Input(784),
            layers.Relu(500),
            layers.Relu(300),
            layers.Softmax(10),
        ],
        loss='categorical_crossentropy',
    )

Also, it's possible to create custom loss functions. Loss function should have two mandatory arguments, namely expected and predicted values.

.. code-block:: python

    import tensorflow as tf
    from neupy import algorithms, layers

    def mean_absolute_error(expected, predicted):
        abs_errors = tf.abs(expected - predicted)
        return tf.reduce_mean(abs_errors)

    nnet = algorithms.GradientDescent(
        [
            layers.Input(784),
            layers.Relu(500),
            layers.Relu(300),
            layers.Softmax(10),
        ],
        loss=mean_absolute_error,
    )

Loss function should return a scalar, because during the training output from the loss function will be used as a variable with respect to which we are differentiating.
