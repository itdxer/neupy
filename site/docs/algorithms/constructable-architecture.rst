Algorithms with Constructable Network Architecture
==================================================

.. contents::

Set up connections
------------------

There are three ways to set up your connection between layers. First one is the simplest one. You just define a list or tuple with the integers. Each integer in the sequence identifies layer's size.

.. code-block:: python

    from neupy import algorithms
    bpnet = algorithms.GradientDescent((2, 4, 1))

In that way we don't actually set up any layer types. By default NeuPy constructs from the tuple simple MLP networks that contains dense layers with sigmoid as a nonlinear activation function.

The second method is the most common one. Instead of defining connections using ``layers.join`` funtion we can simply define it as a parameter to the training algorithm.

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

Also network accepts connections as an input

.. code-block:: python

    from neupy import algorithms
    from neupy.layers import *

    bpnet = algorithms.GradientDescent(
        Input(10) > Sigmoid(40) > Sigmoid(2),
        step=0.2,
        shuffle_data=True
    )

Or we can use predefined connection with ``layers.join`` function.

.. code-block:: python

    from neupy import algorithms, layers

    connection = layers.join(
        layers.Input(10),
        layers.Sigmoid(40),
        layers.Sigmoid(2),
    )

    bpnet = algorithms.GradientDescent(
        connection,
        step=0.2,
        shuffle_data=True
    )

The nice thing that we don't need to initialize connections. Network will do it for you. In addition network compiles training and prediction functions after initializations, which simplifies overall procedrure.

Algorithms
----------

NeuPy supports lots of different training algorithms. You can check :ref:`Cheat sheet <cheatsheet-backprop-algorithms>` if you want to learn more about them.

Each algorithm has a specific set of parameters and not all of the algorithms are sutable for deep learning models. Some of the methods like :network:`Levenberg-Marquardt <LevenbergMarquardt>` or :network:`Conjugate Gradient <ConjugateGradient>` work better for small networks and they would be extremely slow for deep networks. In addtion some of the algorithm are not able to train on mini-batches, so you need to check whether algorithm support mini-batches before using it. Algorithm that support mini-batch training should have ``batch_size`` parameters.

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
    )

Error functions
---------------

NeuPy has many different :ref:`error functions <cheatsheet-error-function>`. You can use different error functions for different problem. For instance, we can use cross entropy for our previous architecture.

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

In addition you can create custom functions. Function suppose to accept two mandatory arguments and return scalar.

.. code-block:: python

    import theano.tensor as T
    from neupy import algorithms, layers

    def mean_absolute_error(expected, predicted):
        return T.abs_(expected - predicted).mean()

    nnet = algorithms.MinibatchGradientDescent(
        [
            layers.Input(784),
            layers.Relu(500),
            layers.Relu(300),
            layers.Softmax(10),
        ],
        error=mean_absolute_error,
    )

Add-ons
-------

Algorithms with constructuble architectures allow to use additional update rules for parameter regularization and step update. For instance, we want to add :network:`Weight Decay <WeightDecay>` regularization and we want to minimize step monotonically after each epoch.

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

Both :network:`WeightDecay` and :network:`StepDecay` algorithms have additional parameters. In case if you need to modify them - you can add them to the training algorithm.

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
