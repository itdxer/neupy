.. _constructible-architecture:

Algorithms with constructible architecture
==========================================

.. contents::

Specify network structure
-------------------------

There are three ways to define relations between layers inside the algorithm. We can use define network architecture separately from the training algorithm.

.. code-block:: python

    from neupy import algorithms, layers

    network = layers.join(
        layers.Input(10),
        layers.Sigmoid(40),
        layers.Sigmoid(2),
    )

    bpnet = algorithms.GradientDescent(
        network,
        step=0.2,
        shuffle_data=True
    )

Or we can set up a list of layers that define sequential relations between layers.

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

This is just a syntax simplification that allows avoiding ``layer.join`` function usage.

Small networks can be defined with a help of inline operator.

.. code-block:: python

    from neupy import algorithms
    from neupy.layers import *

    bpnet = algorithms.GradientDescent(
        Input(10) > Sigmoid(40) > Sigmoid(2),
        step=0.2,
        shuffle_data=True
    )

Train networks with multiple inputs
-----------------------------------

NeuPy allows to train networks with multiple inputs.

.. code-block:: python

    from neupy import algorithms, layers

    gdnet = algorithms.GradientDescent(
        [
            [[
                # 3 categorical inputs
                layers.Input(3),
                layers.Embedding(n_unique_categories, 4),
                layers.Reshape(),
            ], [
                # 17 numerical inputs
                layers.Input(17),
            ]],
            layers.Concatenate(),
            layers.Relu(16),
            layers.Sigmoid(1)
        ],

        step=0.5,
        verbose=True,
        error='binary_crossentropy',
    )

    x_train_cat, x_train_num, y_train = load_train_data()
    x_test_cat, x_test_num, y_test = load_test_data()

    # Categorical variable should be the first, because
    # categorical input layer was defined first in the network
    network.train([x_train_cat, x_train_num], y_train,
                  [x_test_cat, x_test_num], y_test,
                  epochs=180)
    y_predicted = network.predict([x_test_cat, x_test_num])

From the example above you can see that we specified first layer as a list of lists. Each list has small sequence of layers specified and each sequence starts with the ``Input`` layer. This list of lists is just simple syntax sugar around the ``parallel`` function. Exactly the same architecture can be rewritten in the following way.

.. code-block:: python

    gdnet = algorithms.GradientDescent(
        [
            layers.parallel([
                # 3 categorical inputs
                layers.Input(3),
                layers.Embedding(n_unique_categories, 4),
                layers.Reshape(),
            ], [
                # 17 numerical inputs
                layers.Input(17),
            ]),
            layers.Concatenate(),
            layers.Relu(16),
            layers.Sigmoid(1)
        ]
    )

The training nad prediction looks slightly different as well.

.. code-block:: python

    network.train([x_train_cat, x_train_num], y_train,
                  [x_test_cat, x_test_num], y_test,
                  epochs=180)
    y_predicted = network.predict([x_test_cat, x_test_num])

Input we specify as a list where number of values equal to the number of input layers in the network. The order in the list is also important. We defined first input layer for categorical variables therefore we need to pass it as the first element of the input list. The same is true for the ``predict`` method.

Algorithms
----------

NeuPy supports lots of different training algorithms based on the backpropagation. You can check :ref:`Cheat sheet <cheatsheet-backprop-algorithms>` if you want to learn more about them.

Before using these algorithms you must understand that not all of them are suitable for all problems. Some of the methods like :network:`Levenberg-Marquardt <LevenbergMarquardt>` or :network:`Conjugate Gradient <ConjugateGradient>` work better for small networks and they would be extremely slow for networks with millions of parameters. In addition, it's important to note that not all algorithms are possible to train with mini-batches. Algorithms like :network:`Conjugate Gradient <ConjugateGradient>` doesn't work with mini-batches.

Error functions
---------------

NeuPy has many different :ref:`error functions <cheatsheet-error-function>`.

.. code-block:: python

    from neupy import algorithms, layers

    nnet = algorithms.MinibatchGradientDescent(
        [
            layers.Input(784),
            layers.Relu(500),
            layers.Relu(300),
            layers.Softmax(10),
        ],
        error='categorical_crossentropy',
    )

Also, it's possible to create custom error functions. Error function should have two mandatory arguments.

.. code-block:: python

    import tensorflow as tf
    from neupy import algorithms, layers

    def mean_absolute_error(expected, predicted):
        return tf.abs(expected - predicted).mean()

    nnet = algorithms.MinibatchGradientDescent(
        [
            layers.Input(784),
            layers.Relu(500),
            layers.Relu(300),
            layers.Softmax(10),
        ],
        error=mean_absolute_error,
    )

Error function should return a scalar because during the training output from the error function will be used as a variable with respect to which we are differentiating

Add-ons
-------

Algorithms with constructible architectures allow to use additional update rules for parameter regularization and step update. For instance, we want to add :network:`Weight Decay <WeightDecay>` regularization and we want to minimize step monotonically after each epoch.

.. code-block:: python

    from neupy import algorithms, layers

    nnet = algorithms.MinibatchGradientDescent(
        [
            layers.Input(784),
            layers.Relu(500),
            layers.Relu(300),
            layers.Softmax(10),
        ],
        step=0.1,
        batch_size=16,

        addons=[algorithms.WeightDecay,
                algorithms.StepDecay]
    )

Both :network:`WeightDecay` and :network:`StepDecay` algorithms have additional parameters. In case if we need to modify them we can add them to the training algorithm.

.. code-block:: python

    from neupy import algorithms, layers

    nnet = algorithms.MinibatchGradientDescent(
        [
            layers.Input(784),
            layers.Relu(500),
            layers.Relu(300),
            layers.Softmax(10),
        ],

        # Parameters from MinibatchGradientDescent
        step=0.1,
        batch_size=16,

        # Parameters from StepDecay
        reduction_freq=50,

        # Parameters from WeightDecay
        decay_rate=0.05,

        addons=[algorithms.WeightDecay,
                algorithms.StepDecay]
    )

NeuPy doesn't allow to use multiple regularizations and step update add-ons for training algorithm.

.. code-block:: python

    >>> from neupy import algorithms, layers
    >>>
    >>> nnet = algorithms.MinibatchGradientDescent(
    ...     [
    ...         layers.Input(784),
    ...         layers.Relu(500),
    ...         layers.Relu(300),
    ...         layers.Softmax(10),
    ...     ],
    ...     addons=[
    ...         algorithms.WeightDecay,
    ...         algorithms.WeightElimination,
    ...     ]
    ... )
    Traceback (most recent call last):

    ValueError: There can be only one add-on class with type 'Regularization'
